package worker_manager

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"text/template"
	"time"

	"github.com/aifoundry-org/load-balancer/request_router"

	corev1 "k8s.io/api/core/v1"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"sigs.k8s.io/yaml"
)

type WorkerManagerAPI struct {
	router *request_router.RequestRouter
}

func NewWorkerManagerAPI(
	router *request_router.RequestRouter,
) *WorkerManagerAPI {
	return &WorkerManagerAPI{router: router}
}

func getKubeConfig() (*rest.Config, error) {
	config, err := rest.InClusterConfig()
	if err != nil {
		config, err = clientcmd.BuildConfigFromFlags(
			"",
			clientcmd.RecommendedHomeFile,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to get Kubernetes config: %v", err)
		}
	}
	return config, nil
}

func getPodIPByLabelAndNode(
	clientset *kubernetes.Clientset,
	label string,
	nodeName string,
) (string, error) {
	namespace := "default"
	pods, err := clientset.CoreV1().
		Pods(namespace).
		List(context.TODO(), v1.ListOptions{
			LabelSelector: fmt.Sprintf("app=%s", label),
		})
	if err != nil {
		return "", err
	}

	for _, pod := range pods.Items {
		if pod.Spec.NodeName == nodeName {
			return pod.Status.PodIP, nil
		}
	}

	return "", fmt.Errorf(
		"no pod found with label %s on node %s",
		label,
		nodeName,
	)
}

// PostRequest sends a POST request to the specified IP and port with the provided JSON payload.
func PostRequest(
	ip string,
	port int,
	payload map[string]any,
) (map[string]any, error) {
	// Construct the URL
	// TODO: give as argument
	url := fmt.Sprintf("http://%s:%d/content/", ip, port)

	// Marshal the payload into JSON format
	jsonData, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("error marshalling JSON: %w", err)
	}

	// Create a new POST request
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("error creating request: %w", err)
	}

	// Set the content type
	req.Header.Set("Content-Type", "application/json")

	// Create an HTTP client and make the request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("error making request: %w", err)
	}
	defer resp.Body.Close()

	// Read the response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("error reading response body: %w", err)
	}

	// Unmarshal the response body into a map
	var response map[string]any
	err = json.Unmarshal(body, &response)
	if err != nil {
		return nil, fmt.Errorf("error unmarshalling response: %w", err)
	}

	return response, nil
}

// PodTemplateData holds the data to be injected into the pod template
type PodTemplateData struct {
	NodeName   string
	ModelAlias string
	ModelURL   string
	ModelPath  string
}

// createWorkerPodManifest creates a pod manifest for a worker with the given parameters
// using Go templates to process the template file
func createWorkerPodManifest(
	templatePath string,
	nodeName string,
	modelAlias string,
	modelURL string,
	modelPath string,
) (*corev1.Pod, error) {
	// Read the template file
	yamlFile, err := os.ReadFile(templatePath)
	if err != nil {
		return nil, fmt.Errorf("error reading template file: %w", err)
	}

	// Prepare template data
	templateData := PodTemplateData{
		NodeName:   nodeName,
		ModelAlias: modelAlias,
		ModelURL:   modelURL,
		ModelPath:  modelPath,
	}

	// Parse and execute the template
	tmpl, err := template.New("pod-template").Parse(string(yamlFile))
	if err != nil {
		return nil, fmt.Errorf("error parsing template: %w", err)
	}

	var processedTemplate bytes.Buffer
	if err := tmpl.Execute(&processedTemplate, templateData); err != nil {
		return nil, fmt.Errorf("error executing template: %w", err)
	}

	var pod corev1.Pod
	if err := yaml.Unmarshal(processedTemplate.Bytes(), &pod); err != nil {
		return nil, fmt.Errorf("error decoding pod manifest: %w", err)
	}

	// Set a unique name for the pod based on model alias and a timestamp
	pod.ObjectMeta.Name = fmt.Sprintf("worker-%s-%d",
		sanitizeName(modelAlias),
		time.Now().Unix())

	return &pod, nil
}

func (self *WorkerManagerAPI) addWorker(
	nodeName string,
	modelAlias string,
	modelURL string,
	credentials string,
) (*request_router.Worker, error) {

	fmt.Printf("addWorker %s\n", nodeName)

	if nodeName == "" {
		return nil, fmt.Errorf("nodeName can't be empty")
	}
	if modelAlias == "" {
		return nil, fmt.Errorf("modelAlias can't be empty")
	}
	if modelURL == "" {
		return nil, fmt.Errorf("modelUrl can't be empty")
	}

	config, err := getKubeConfig()
	if err != nil {
		return nil, fmt.Errorf("error building kubeconfig: %v", err)
	}

	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("error creating kubernetes client: %v", err)
	}

	storageManagerIP, err := getPodIPByLabelAndNode(
		clientset,
		"nekko-sm",
		nodeName,
	)
	if err != nil {
		// TODO: return proper error
		return nil, err
	}

	fmt.Printf("Storage manager IP %v\n", storageManagerIP)

	modelRequest := map[string]any{
		"credentials": credentials,
		"url":         modelURL,
	}

	resp, err := PostRequest(storageManagerIP, 8050, modelRequest)
	if err != nil {
		// TODO: return proper error
		return nil, err
	}

	modelDigest := resp["digest"]

	// Extract the digest string and construct the model path
	digestStr, ok := modelDigest.(string)
	if !ok {
		return nil, fmt.Errorf("invalid digest format: %v", modelDigest)
	}

	// Construct path according to the format in the TODO comment
	// Split the digest to insert the additional directory level
	parts := bytes.Split([]byte(digestStr), []byte(":"))
	if len(parts) != 2 {
		return nil, fmt.Errorf(
			"invalid digest format, expected 'algo:hash', got: %s",
			digestStr,
		)
	}

	algo := string(parts[0])
	hash := string(parts[1])

	// Enforce SHA256 algorithm
	if algo != "sha256" {
		return nil, fmt.Errorf(
			"unsupported digest algorithm: %s, only sha256 is supported",
			algo,
		)
	}

	// Validate hash to prevent path traversal attacks
	// SHA256 hashes are 64 hex characters
	if len(hash) != 64 {
		return nil, fmt.Errorf(
			"invalid hash length: expected 64 characters for SHA256",
		)
	}

	// Ensure hash only contains valid hexadecimal characters
	for _, c := range hash {
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
			return nil, fmt.Errorf("invalid character in hash: %c", c)
		}
	}

	modelPath := fmt.Sprintf("/var/lib/nekko/cache/blobs/%s/%s", algo, hash)

	// Create the Pod manifest from template
	templatePath := "/app/templates/worker-pod-template.yaml"
	pod, err := createWorkerPodManifest(
		templatePath,
		nodeName,
		modelAlias,
		modelURL,
		modelPath,
	)
	if err != nil {
		return nil, fmt.Errorf("error creating pod manifest: %w", err)
	}

	fmt.Printf("Creating pod %s in namespace %s\n", pod.Name, pod.Namespace)

	// TODO: remove. Ensure the pod is created in the default namespace where our service account has permissions
	if pod.Namespace == "" {
		pod.Namespace = "default"
		fmt.Println("Setting namespace to default")
	}

	// Deploy the Pod
	createdPod, err := clientset.CoreV1().
		Pods(pod.Namespace).
		Create(context.Background(), pod, v1.CreateOptions{})
	if err != nil {
		return nil, fmt.Errorf("error creating pod: %w", err)
	}

	// Wait for the pod to get an IP address
	podIP := ""
	maxRetries := 30
	retryInterval := 2 * time.Second

	fmt.Printf("Waiting for pod %s to get an IP address...\n", createdPod.Name)

	for i := range maxRetries {
		// Get the latest pod information
		pod, err := clientset.CoreV1().
			Pods(createdPod.Namespace).
			Get(context.Background(), createdPod.Name, v1.GetOptions{})
		if err != nil {
			return nil, fmt.Errorf("error getting pod status: %w", err)
		}

		// Check if the pod has an IP address
		if pod.Status.PodIP != "" {
			podIP = pod.Status.PodIP
			fmt.Printf("Pod %s has IP address: %s\n", pod.Name, podIP)
			break
		}

		fmt.Printf(
			"Retry %d/%d: Pod %s does not have an IP address yet\n",
			i+1,
			maxRetries,
			pod.Name,
		)
		time.Sleep(retryInterval)
	}

	if podIP == "" {
		return nil, fmt.Errorf(
			"timed out waiting for pod %s to get an IP address",
			createdPod.Name,
		)
	}

	// Construct the worker URL using the pod IP and the expected port
	workerURL := fmt.Sprintf("http://%s:8000", podIP)

	model := request_router.Model{
		FullName: modelAlias, // Set the full name to the model URL
		Alias:    modelAlias,
		Metadata: make(map[string]interface{}),
	}

	return &request_router.Worker{
		URL:   workerURL,
		Model: model,
	}, nil
}

type PostRequestBody struct {
	ModelURL    string `json:"modelUrl"`
	ModelAlias  string `json:"modelAlias"`
	NodeName    string `json:"nodeName"`
	Credentials string `json:"credentials"`
}

func (self *WorkerManagerAPI) handlePost(
	w http.ResponseWriter,
	r *http.Request,
) {
	// POST params:
	//   node_name: to be used in a selector
	//   model: url accepted by storage_manager
	//   model_alias: the alias for the model
	// - Donwloads particular model using storage manager on the node.
	// - Starts a pod with downloaded model configured.
	// - Sets label `model_alias` on a pod to provided `model_alias`.
	// - Adds worker with corresponding alias and pod ip to the router.
	//
	var reqBody PostRequestBody
	decoder := json.NewDecoder(r.Body)
	err := decoder.Decode(&reqBody)
	if err != nil {
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}

	// Process the parsed data (reqBody)
	worker, err := self.addWorker(
		reqBody.NodeName,
		reqBody.ModelAlias,
		reqBody.ModelURL,
		reqBody.Credentials,
	)

	if err != nil {
		fmt.Printf("Error %v\n", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	self.router.AddWorker(*worker)

	// Example response
	w.WriteHeader(http.StatusOK)

}

// sanitizeName converts a string to a valid Kubernetes resource name
func sanitizeName(name string) string {
	// Replace dots and other invalid characters with hyphens
	// and convert to lowercase for Kubernetes naming conventions
	result := ""
	for _, r := range name {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '-' {
			result += string(r)
		} else if r >= 'A' && r <= 'Z' {
			// Convert uppercase to lowercase
			result += string(r - 'A' + 'a')
		} else {
			// Replace invalid chars with hyphen
			result += "-"
		}
	}
	return result
}

func (self *WorkerManagerAPI) handleDelete(
	w http.ResponseWriter,
	r *http.Request,
) {
	// Extract worker name from query parameters
	workerName := r.URL.Query().Get("name")
	if workerName == "" {
		http.Error(w, "Worker name is required", http.StatusBadRequest)
		return
	}

	// Get Kubernetes client
	config, err := getKubeConfig()
	if err != nil {
		http.Error(
			w,
			fmt.Sprintf("Error building kubeconfig: %v", err),
			http.StatusInternalServerError,
		)
		return
	}

	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		http.Error(
			w,
			fmt.Sprintf("Error creating kubernetes client: %v", err),
			http.StatusInternalServerError,
		)
		return
	}

	// Delete the pod
	namespace := "default" // Using default namespace
	err = clientset.CoreV1().
		Pods(namespace).
		Delete(context.Background(), workerName, v1.DeleteOptions{})
	if err != nil {
		http.Error(
			w,
			fmt.Sprintf("Error deleting pod: %v", err),
			http.StatusInternalServerError,
		)
		return
	}

	// Remove the worker from the router
	deleted := self.router.DeleteWorkerByName(workerName)

	// Return success response
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	fmt.Fprintf(
		w,
		`{"status":"success","message":"Worker %s deleted","found_in_router":%t}`,
		workerName,
		deleted,
	)
}

func (self *WorkerManagerAPI) ServeHTTP(
	w http.ResponseWriter,
	r *http.Request,
) {
	fmt.Println("Hello from worker manager.")
	w.Header().Set("Content-Type", "application/json")

	switch r.Method {
	// case http.MethodGet:
	// self.handleGet(rw, req)
	case http.MethodPost:
		self.handlePost(w, r)
	case http.MethodDelete:
		self.handleDelete(w, r)
	default:
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
	}

	// TODO: enforce same alias to correspond the same model.
	// TODO: deploy on multiple nodes at once.
	// TODO: can we have multiple of the same model aliases
	//       on the same node? Of course.
	// DELETE params:
	//   mode_name: to be used in selector
	//   model_alias: to select pod by the label
	// - Removes worker from the router.
	// - Removes the pod from the node.
}
