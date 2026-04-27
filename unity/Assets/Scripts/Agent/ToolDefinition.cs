using System;
using System.Collections.Generic;
using UnityEngine;

namespace NPCAI.Agent
{
    /// <summary>
    /// Tool category — DE skill or VN game-engine action.
    /// </summary>
    public enum ToolCategory
    {
        Skill,           // DE 24 internal skills (Logic, Empathy, ...)
        Character,       // set_expression, show_character, hide_character
        Scene,           // set_background, show_cg
        Audio,           // play_bgm, play_sfx
        Flow,            // present_choices, narrate, end_scene
        Meta,            // remember, recall
    }

    /// <summary>
    /// Static description of one tool the agent can call.
    /// Loaded into the system prompt so the LLM knows the schema.
    /// </summary>
    [Serializable]
    public class ToolDefinition
    {
        public string name;
        public ToolCategory category;
        public string description;
        public List<ToolParam> parameters = new List<ToolParam>();

        public string ToPromptSchema()
        {
            var sb = new System.Text.StringBuilder();
            sb.Append($"- {name}");
            if (parameters.Count > 0)
            {
                sb.Append("(");
                for (int i = 0; i < parameters.Count; i++)
                {
                    if (i > 0) sb.Append(", ");
                    sb.Append(parameters[i].name);
                    sb.Append(": ");
                    sb.Append(parameters[i].type);
                    if (parameters[i].required) sb.Append("*");
                }
                sb.Append(")");
            }
            sb.Append($" — {description}");
            return sb.ToString();
        }
    }

    [Serializable]
    public class ToolParam
    {
        public string name;
        public string type;          // "string", "int", "float", "bool", "string[]"
        public bool required = true;
        public string description;
        public string[] enumValues;  // for enum-typed params (e.g., emotion = {happy,sad,...})
    }

    /// <summary>
    /// Runtime tool call produced by the LLM.
    /// </summary>
    [Serializable]
    public class ToolCall
    {
        public string name;
        public Dictionary<string, object> args = new Dictionary<string, object>();

        public T Get<T>(string key, T fallback = default)
        {
            if (!args.ContainsKey(key)) return fallback;
            try { return (T)Convert.ChangeType(args[key], typeof(T)); }
            catch { return fallback; }
        }

        public string[] GetStringArray(string key)
        {
            if (!args.ContainsKey(key)) return new string[0];
            if (args[key] is string[] arr) return arr;
            if (args[key] is List<object> list)
            {
                var result = new string[list.Count];
                for (int i = 0; i < list.Count; i++) result[i] = list[i]?.ToString() ?? "";
                return result;
            }
            return new string[0];
        }
    }
}
