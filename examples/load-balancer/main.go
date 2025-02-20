package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httputil"
	"net/url"
	"request_router/request_router"
	"strconv"
	"strings"
)

type Proxy struct {
	router *request_router.RequestRouter
}

func NewProxy(router *request_router.RequestRouter) *Proxy {
	return &Proxy{router: router}
}

func modifyResponse(router *request_router.RequestRouter, recordId string) func(*http.Response) error {
	// Create a function closure for modifying JSON
	jsonModifier := func(jsonData map[string]interface{}) {
		modifyJSONData(jsonData)
	}

	finishCallback := func(responseMessage map[string]interface{}, e error) {
		if e != nil {
			router.AddRequestError(recordId, e)
		} else {
			router.AddRequestFinish(recordId, responseMessage)
		}
	}

	return func(resp *http.Response) error {
		// Detect SSE by checking the Content-Type header
		if strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream") {
			// Modify SSE stream
			reader, writer := io.Pipe()
			go modifySSEStream(resp.Body, writer, jsonModifier, finishCallback)
			resp.Body = reader
			return nil
		}

		// Otherwise, handle as regular JSON
		return modifyJSONResponse(resp, jsonModifier, finishCallback)
	}
}

// modifyJSONResponse modifies a standard JSON response using a provided modification function.
func modifyJSONResponse(resp *http.Response, jsonModifier func(map[string]interface{}), finishCallback func(map[string]interface{}, error)) error {
	// Read the original response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		resp.Body.Close()
		finishCallback(nil, err)
		return err
	}
	// defer resp.Body.Close()

	// Parse JSON
	var jsonData map[string]interface{}
	if err := json.Unmarshal(body, &jsonData); err != nil {
		resp.Body.Close()
		finishCallback(nil, err)
		return err
	}

	// Apply the provided JSON modification function
	jsonModifier(jsonData)

	// Marshal back to JSON
	modifiedBody, err := json.Marshal(jsonData)
	if err != nil {
		resp.Body.Close()
		finishCallback(nil, err)
		return err
	}

	// Update response body with modified JSON
	resp.Body = io.NopCloser(bytes.NewReader(modifiedBody))
	resp.ContentLength = int64(len(modifiedBody))
	resp.Header.Set("Content-Length", strconv.Itoa(len(modifiedBody)))

	choices, _ := jsonData["choices"].([]interface{})
	choice0, _ := choices[0].(map[string]interface{})
	message, _ := choice0["message"].(map[string]interface{})
	responseMessage := message

	finishCallback(responseMessage, nil)

	return nil
}

// modifySSEStream modifies SSE events containing JSON using a provided modification function.
func modifySSEStream(input io.ReadCloser, output *io.PipeWriter, jsonModifier func(map[string]interface{}), finishCallback func(map[string]interface{}, error)) {
	var responseMessage map[string]interface{} = make(map[string]interface{})
	defer input.Close()
	defer output.Close()

	scanner := bufio.NewScanner(input)
	for scanner.Scan() {
		line := scanner.Text()

		// Check if it's an SSE data line
		if strings.HasPrefix(line, "data: ") {
			jsonStr := strings.TrimPrefix(line, "data: ")

			// Attempt to parse JSON
			var jsonData map[string]interface{}
			if err := json.Unmarshal([]byte(jsonStr), &jsonData); err == nil {
				// Apply the provided JSON modification function
				jsonModifier(jsonData)

				// TODO: extract and build messages

				// Marshal back to JSON
				modifiedJSON, _ := json.Marshal(jsonData)

				choices, _ := jsonData["choices"].([]interface{})
				choice0, _ := choices[0].(map[string]interface{})
				delta, _ := choice0["delta"].(map[string]interface{})
				role, ok := delta["role"].(string)
				if ok {
					responseMessage["role"] = role
				}
				content, ok := delta["content"].(string)
				if ok {

					oldContent, ok := responseMessage["content"]

					if ok {
						newContent := fmt.Sprintf("%s%s", oldContent, content)
						responseMessage["content"] = newContent
					} else {
						responseMessage["content"] = content
					}
				}

				// Send modified event
				line = "data: " + string(modifiedJSON)
			}
		}

		// Write the modified (or unmodified) line to the output stream
		_, _ = output.Write([]byte(line + "\n"))
	}

	if err := scanner.Err(); err != nil {
		finishCallback(nil, err)
		_ = output.CloseWithError(err)
	}

	finishCallback(responseMessage, nil)
}

// modifyJSONData modifies a JSON object by changing the specified field.
func modifyJSONData(jsonData map[string]interface{}) {
	// TODO
	// Replace full model name with alias.
	jsonData["model"] = "kvarbakulis"
}

func modifyRequest(proxy *httputil.ReverseProxy, router *request_router.RequestRouter, req *http.Request) error {
	// Read the original body
	body, err := io.ReadAll(req.Body)
	if err != nil {
		return err
	}
	defer req.Body.Close() // Close original body

	// Parse JSON
	var jsonData map[string]interface{}
	if err := json.Unmarshal(body, &jsonData); err != nil {
		return err
	}

	// Set target URL for the proxy.
	target_worker, _ := router.ChooseWorker(jsonData)
	target_url, _ := url.Parse(target_worker.URL)

	req.URL.Scheme = target_url.Scheme
	req.URL.Host = target_url.Host

	recordId := router.AddRequestStart(target_worker.URL, target_worker.Model.Alias, jsonData)

	// Modify the model field from alias to the full model name.
	jsonData["model"] = target_worker.Model.FullName

	// Marshal back to JSON
	modifiedBody, err := json.Marshal(jsonData)
	if err != nil {
		return err
	}

	// Update request Body with modified JSON
	req.Body = io.NopCloser(bytes.NewReader(modifiedBody))
	req.ContentLength = int64(len(modifiedBody))

	proxy.ModifyResponse = modifyResponse(router, recordId)

	return nil
}

func (p *Proxy) ServeHTTP(rw http.ResponseWriter, req *http.Request) {
	proxy := httputil.ReverseProxy{}
	proxy.Director = func(req *http.Request) {
		modifyRequest(&proxy, p.router, req)
	}
	// proxy.ModifyResponse = modifyResponse()

	proxy.ServeHTTP(rw, req)
}

func v1Models(router *request_router.RequestRouter) func(http.ResponseWriter, *http.Request) {

	return func(rw http.ResponseWriter, req *http.Request) {
		rw.Header().Set("Content-Type", "application/json")

		models := router.ModelAliases()

		// Initialize the top-level JSON structure
		response := map[string]interface{}{
			"object": "list",
			"data":   []interface{}{},
		}

		for _, model := range models {
			// Create the model entry as a map
			modelEntry := map[string]interface{}{
				"id":          model,
				"object":      "model",
				"owned_by":    "user",
				"permissions": []string{},
			}

			// Append the model entry to the "data" array
			response["data"] = append(response["data"].([]interface{}), modelEntry)
		}

		encoder := json.NewEncoder(rw)
		encoder.Encode(response)
	}
}

func main() {
	// Manual seed for testing.
	// model_llama := request_router.Model{
	// 	FullName: "llama",
	// 	Alias:    "llama",
	// }
	// worker := request_router.Worker{
	// 	URL:   "http://localhost:8000",
	// 	Model: model_llama,
	// }
	router := request_router.NewRequestRouter([]request_router.Worker{})
	router.SeedWorkersFromK8s()

	proxy := NewProxy(router)

	mux := http.NewServeMux()

	mux.Handle("/", proxy)

	mux.HandleFunc("/v1/models", v1Models(router))
	// TODO: add function to handle /v1/models

	// TODO: refactor into proxy mod.
	http.ListenAndServe(":8080", mux)
}
