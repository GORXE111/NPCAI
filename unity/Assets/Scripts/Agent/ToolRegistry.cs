using System;
using System.Collections.Generic;
using UnityEngine;

namespace NPCAI.Agent
{
    /// <summary>
    /// Central registry of all tools the agent can call.
    /// - Schema for LLM prompt
    /// - Dispatcher for runtime execution against Unity systems
    ///
    /// Tools are registered with a delegate (handler) at startup by VisualNovelManager.
    /// </summary>
    public class ToolRegistry
    {
        // Singleton-ish access (set by VisualNovelManager on Awake)
        public static ToolRegistry Instance { get; private set; }

        public delegate void ToolHandler(ToolCall call, Action<string> onComplete);

        private readonly Dictionary<string, ToolDefinition> defs = new Dictionary<string, ToolDefinition>();
        private readonly Dictionary<string, ToolHandler> handlers = new Dictionary<string, ToolHandler>();

        public ToolRegistry()
        {
            Instance = this;
        }

        public void Register(ToolDefinition def, ToolHandler handler)
        {
            if (def == null || string.IsNullOrEmpty(def.name))
            {
                Debug.LogError("[ToolRegistry] Cannot register tool with null/empty name");
                return;
            }
            defs[def.name] = def;
            handlers[def.name] = handler;
        }

        public bool Has(string name) => defs.ContainsKey(name);

        public ToolDefinition GetDef(string name) => defs.TryGetValue(name, out var d) ? d : null;

        public IEnumerable<ToolDefinition> AllDefs() => defs.Values;

        /// <summary>
        /// Build the schema block to inject into the LLM system prompt.
        /// </summary>
        public string BuildPromptSchema()
        {
            var sb = new System.Text.StringBuilder();
            sb.AppendLine("Available tools (call by emitting JSON in tool_calls field):");

            // Group by category for readability
            var byCat = new Dictionary<ToolCategory, List<ToolDefinition>>();
            foreach (var d in defs.Values)
            {
                if (!byCat.ContainsKey(d.category)) byCat[d.category] = new List<ToolDefinition>();
                byCat[d.category].Add(d);
            }
            foreach (var cat in byCat.Keys)
            {
                sb.AppendLine($"\n[{cat}]");
                foreach (var d in byCat[cat])
                {
                    sb.AppendLine("  " + d.ToPromptSchema());
                }
            }
            return sb.ToString();
        }

        /// <summary>
        /// Execute a tool call. Calls handler synchronously; handler may complete async via callback.
        /// </summary>
        public void Execute(ToolCall call, Action<string> onComplete)
        {
            if (call == null || string.IsNullOrEmpty(call.name))
            {
                onComplete?.Invoke("[tool_error: empty call]");
                return;
            }
            if (!handlers.ContainsKey(call.name))
            {
                Debug.LogWarning($"[ToolRegistry] Unknown tool: {call.name}");
                onComplete?.Invoke($"[tool_error: unknown tool '{call.name}']");
                return;
            }

            try
            {
                handlers[call.name](call, onComplete);
            }
            catch (Exception ex)
            {
                Debug.LogError($"[ToolRegistry] Tool '{call.name}' threw: {ex.Message}");
                onComplete?.Invoke($"[tool_error: {ex.Message}]");
            }
        }

        public int Count => defs.Count;
    }
}
