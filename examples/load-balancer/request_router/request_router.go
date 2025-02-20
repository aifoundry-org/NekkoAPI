package request_router

import (
	"context"
	"errors"
	"fmt"
	"log"
	"math/rand"
	"sync"
	"time"

	"github.com/google/uuid"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
)

type RequestRecord struct {
	ID       string
	Query    map[string]interface{} // original JSON query
	Response map[string]interface{} // response message
	Start    time.Time
	Finish   time.Time
	Error    error
}

// Model represents a Language Model with its details.
type Model struct {
	FullName string                 // Full name of the model
	Alias    string                 // Alias for easier reference
	Metadata map[string]interface{} // Additional metadata
}

// Worker represents a worker node that can serve LLM models.
type Worker struct {
	URL     string // URL of the worker node
	Model   Model  // List of deployed LLM models on this node
	BusyIds map[string]bool
	History []RequestRecord
}

// RouterState holds the current state of the router.
type RouterState struct {
	Workers []Worker
	mu      sync.RWMutex // Mutex to protect access to WorkerNodes
}

// RequestRouter handles routing of inference requests to worker nodes.
type RequestRouter struct {
	state *RouterState
	rnd   *rand.Rand
}

// NewRequestRouter initializes a new RequestRouter with the provided worker nodes.
func NewRequestRouter(workerNodes []Worker) *RequestRouter {
	src := rand.NewSource(time.Now().UnixNano())
	rnd := rand.New(src)

	return &RequestRouter{
		state: &RouterState{
			Workers: workerNodes,
		},
		rnd: rnd,
	}
}

func (rr *RequestRouter) ModelAliases() []string {
	var aliasSet map[string]bool = make(map[string]bool)
	var result []string

	for _, worker := range rr.state.Workers {
		aliasSet[worker.Model.Alias] = true
	}
	for key, _ := range aliasSet {
		result = append(result, key)
	}

	return result
}

func (w *Worker) IsBusy() bool {
	return w.BusyLoad() != 0
}

func (w *Worker) BusyLoad() int {
	if w.BusyIds == nil {
		return 0
	}

	return len(w.BusyIds)
}

func (w *Worker) AddBusy(recordId string) {
	if w.BusyIds == nil {
		w.BusyIds = make(map[string]bool)
	}

	w.BusyIds[recordId] = true
}

func (w *Worker) RemoveBusy(recordId string) {
	if w.BusyIds == nil {
		w.BusyIds = make(map[string]bool)
	}

	delete(w.BusyIds, recordId)
}

// ChooseWorker selects a worker node based on the routing logic.
// Currently, it selects a random worker node that has the required model.
func (rr *RequestRouter) ChooseWorker(req map[string]interface{}) (*Worker, error) {
	rr.state.mu.RLock()
	defer rr.state.mu.RUnlock()

	model_alias := req["model"]

	log.Println("-- Selecting request route --")
	// Filter worker nodes that have the required model
	var eligibleWorkers []Worker
	for _, worker := range rr.state.Workers {
		if worker.Model.Alias == model_alias {
			eligibleWorkers = append(eligibleWorkers, worker)
		}
	}

	if len(eligibleWorkers) == 0 {
		return nil, errors.New("no available workers with the specified model")
	}

	var availableWorkers []Worker
	for _, worker := range eligibleWorkers {
		if worker.BusyLoad() == 0 {
			availableWorkers = append(availableWorkers, worker)
		} else {
			log.Printf("Worker %v queue %v BUSY\n", worker.URL, worker.BusyLoad())
		}
	}

	if len(availableWorkers) == 0 {
		log.Println("All BUSY")
		availableWorkers = eligibleWorkers
	}

	// Find the longest prefix
	messages, _ := req["messages"].([]interface{})
	reqPromptish := messagesPromptish(messages)
	var availablePromptish []string
	for _, worker := range availableWorkers {
		history := worker.History
		if len(history) > 0 {
			lastRecord := history[len(history)-1]
			promptish := lastRecord.Promptish()
			log.Printf("Worker %v queue %v kv-match %v\n", worker.URL, worker.BusyLoad(), longestCommonPrefixLength(reqPromptish, promptish))
			availablePromptish = append(availablePromptish, promptish)
		} else {
			log.Printf("Worker %v queue %v kv-match %v\n", worker.URL, worker.BusyLoad(), 0)
			availablePromptish = append(availablePromptish, "")
		}
	}
	maxIndex := findLongestCommonPrefixIndex(availablePromptish, reqPromptish)

	var selectedWorker Worker
	// TODO: instead of simple logic use cost function,
	//       taking into account prompt lenght, kv (promptish)
	//       match, busy estimates (sampling or max_completion_tokens)
	// TODO: Estimate cost approximation function from history
	//       (for individual nodes).

	if maxIndex < 0 {
		// Select a random worker from the available
		selectedWorker = availableWorkers[rr.rnd.Intn(len(availableWorkers))]
	} else {
		log.Printf("Prompt prefix best match %v\n", longestCommonPrefixLength(reqPromptish, availablePromptish[maxIndex]))
		selectedWorker = availableWorkers[maxIndex]
	}

	log.Printf("Selected worker %v load %v\n", selectedWorker.URL, selectedWorker.BusyLoad())
	return &selectedWorker, nil
}

func (rr *RequestRouter) AddRequestStart(url string, model string, query map[string]interface{}) string {
	rr.state.mu.Lock()
	defer rr.state.mu.Unlock()

	start := time.Now()

	recordId := uuid.New().String()

	record := RequestRecord{
		ID:    recordId,
		Query: query,
		Start: start,
	}

	for i, worker := range rr.state.Workers {
		if worker.URL == url && worker.Model.Alias == model {
			rr.state.Workers[i].History = append(worker.History, record)
			rr.state.Workers[i].AddBusy(recordId)
		}
	}

	return recordId
}

func (rr *RequestRouter) AddRequestFinish(recordId string, response map[string]interface{}) {
	rr.state.mu.Lock()
	defer rr.state.mu.Unlock()

	finish := time.Now()

	for i, worker := range rr.state.Workers {
		for j, record := range worker.History {
			if record.ID == recordId {
				rr.state.Workers[i].History[j].Response = response
				rr.state.Workers[i].History[j].Finish = finish
				rr.state.Workers[i].RemoveBusy(recordId)
			}
		}
	}
}

func (rr *RequestRouter) AddRequestError(recordId string, e error) {
	rr.state.mu.Lock()
	defer rr.state.mu.Unlock()

	finish := time.Now()

	for i, worker := range rr.state.Workers {
		for j, record := range worker.History {
			if record.ID == recordId {
				rr.state.Workers[i].History[j].Error = e
				rr.state.Workers[i].History[j].Finish = finish
				rr.state.Workers[i].RemoveBusy(recordId)
			}
		}
	}
}

func (rr *RequestRouter) SeedWorkersFromK8s() {
	urls, _ := getPodURLs("nekko-api", "default", "8000")
	var workers []Worker
	log.Println("Seeding workers...")
	// TODO: fetch model names from the workers?
	for _, url := range urls {
		worker := Worker{
			URL: url,
			Model: Model{
				Alias:    "smollm2",
				FullName: "smollm2",
			},
		}
		workers = append(workers, worker)
		log.Printf("Adding worker %v with model %v\n", worker.URL, worker.Model.Alias)
	}

	rr.state.mu.Lock()
	defer rr.state.mu.Unlock()
	rr.state.Workers = workers
}

func getKubeConfig() (*rest.Config, error) {
	config, err := rest.InClusterConfig()
	if err != nil {
		config, err = clientcmd.BuildConfigFromFlags("", clientcmd.RecommendedHomeFile)
		if err != nil {
			return nil, fmt.Errorf("failed to get Kubernetes config: %v", err)
		}
	}
	return config, nil
}

func getPodURLs(appLabel, namespace, port string) ([]string, error) {
	config, err := getKubeConfig()
	if err != nil {
		return nil, err
	}

	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create clientset: %v", err)
	}

	pods, err := clientset.CoreV1().Pods(namespace).List(context.TODO(), metav1.ListOptions{
		LabelSelector: fmt.Sprintf("app=%s", appLabel),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to list pods: %v", err)
	}

	var urls []string
	for _, pod := range pods.Items {
		if pod.Status.Phase == v1.PodRunning {
			for _, ip := range pod.Status.PodIPs {
				urls = append(urls, fmt.Sprintf("http://%s:%s", ip.IP, port))
			}
		}
	}

	return urls, nil
}

// Return emulated prompt by concatenating request query messages.
// To be used in kv cache heuristics.
func (record *RequestRecord) Promptish() string {
	// TODO: also add response messages
	messages, _ := record.Query["messages"].([]interface{})

	if record.Response != nil {
		messages = append(messages, record.Response)
	}

	return messagesPromptish(messages)
}

func messagesPromptish(messages []interface{}) string {
	var prompt string

	for _, message := range messages {
		entry, _ := message.(map[string]interface{})
		prompt += fmt.Sprintf("<%s>", entry["role"])
		prompt += fmt.Sprintf("%s", entry["content"])
	}

	return prompt
}

// longestCommonPrefixLength returns the length of the longest common prefix between two strings
func longestCommonPrefixLength(a, b string) int {
	length := 0
	for i := 0; i < len(a) && i < len(b); i++ {
		if a[i] != b[i] {
			break
		}
		length++
	}
	return length
}

// findLongestCommonPrefixIndex finds the index of the string in lst that has the longest common prefix with s
func findLongestCommonPrefixIndex(lst []string, s string) int {
	maxIndex := -1
	maxLength := 0

	for i, str := range lst {
		commonLength := longestCommonPrefixLength(str, s)
		if commonLength > maxLength {
			maxLength = commonLength
			maxIndex = i
		}
	}

	return maxIndex
}
