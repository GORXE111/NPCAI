using System;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

namespace NPCLLM.LLM
{
    [Serializable]
    public class ChatMessage
    {
        public string role;
        public string content;
    }

    [Serializable]
    public class ChatResponse
    {
        public ChatMessage message;
        public bool done;
        public long total_duration;
        public long eval_count;
    }

    public class OllamaClient : MonoBehaviour
    {
        [Header("Ollama Settings")]
        [SerializeField] private string ollamaUrl = "http://localhost:11434";
        [SerializeField] private string model = "qwen3.5:9b";
        [SerializeField] private float temperature = 0.7f;
        [SerializeField] private int maxTokens = 256;
        [SerializeField] private bool enableThinking = false;

        public string Model => model;

        public void SendChat(List<ChatMessage> messages, Action<string> onSuccess, Action<string> onError)
        {
            // Build JSON manually to support "think" field in options
            string messagesJson = BuildMessagesJson(messages);
            string json = "{" +
                "\"model\":\"" + EscapeJson(model) + "\"," +
                "\"messages\":" + messagesJson + "," +
                "\"stream\":false," +
                "\"options\":{" +
                    "\"temperature\":" + temperature.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "," +
                    "\"num_predict\":" + maxTokens +
                "}," +
                "\"think\":" + (enableThinking ? "true" : "false") +
            "}";

            StartCoroutine(PostRequest("/api/chat", json, onSuccess, onError));
        }

        private string BuildMessagesJson(List<ChatMessage> messages)
        {
            var sb = new StringBuilder("[");
            for (int i = 0; i < messages.Count; i++)
            {
                if (i > 0) sb.Append(",");
                sb.Append("{\"role\":\"");
                sb.Append(EscapeJson(messages[i].role));
                sb.Append("\",\"content\":\"");
                sb.Append(EscapeJson(messages[i].content));
                sb.Append("\"}");
            }
            sb.Append("]");
            return sb.ToString();
        }

        private static string EscapeJson(string s)
        {
            if (string.IsNullOrEmpty(s)) return "";
            return s.Replace("\\", "\\\\")
                    .Replace("\"", "\\\"")
                    .Replace("\n", "\\n")
                    .Replace("\r", "\\r")
                    .Replace("\t", "\\t");
        }

        private System.Collections.IEnumerator PostRequest(string endpoint, string json, Action<string> onSuccess, Action<string> onError)
        {
            string url = ollamaUrl + endpoint;
            byte[] bodyRaw = Encoding.UTF8.GetBytes(json);

            using var www = new UnityWebRequest(url, "POST");
            www.uploadHandler = new UploadHandlerRaw(bodyRaw);
            www.downloadHandler = new DownloadHandlerBuffer();
            www.SetRequestHeader("Content-Type", "application/json");
            www.timeout = 120;

            yield return www.SendWebRequest();

            if (www.result != UnityWebRequest.Result.Success)
            {
                onError?.Invoke(www.error);
                yield break;
            }

            string responseText = www.downloadHandler.text;
            try
            {
                var response = JsonUtility.FromJson<ChatResponse>(responseText);
                string content = response.message != null ? response.message.content : "";

                // Strip thinking tags if present
                if (!string.IsNullOrEmpty(content))
                {
                    int thinkEnd = content.IndexOf("</think>");
                    if (thinkEnd >= 0)
                        content = content.Substring(thinkEnd + 8).TrimStart();
                }

                onSuccess?.Invoke(content);
            }
            catch (Exception e)
            {
                onError?.Invoke("Parse error: " + e.Message);
            }
        }
    }
}
