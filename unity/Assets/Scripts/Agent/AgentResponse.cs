using System;
using System.Collections.Generic;
using UnityEngine;

namespace NPCAI.Agent
{
    /// <summary>
    /// Parsed LLM output: dialogue text + ordered tool calls.
    /// Expected JSON schema from the LLM:
    /// {
    ///   "dialogue": "the NPC's spoken/written line",
    ///   "tool_calls": [
    ///     {"name": "set_expression", "args": {"actor": "Kim", "emotion": "neutral"}},
    ///     {"name": "present_choices", "args": {"options": ["A","B","C"]}}
    ///   ]
    /// }
    /// </summary>
    [Serializable]
    public class AgentResponse
    {
        public string dialogue;
        public List<ToolCall> toolCalls = new List<ToolCall>();
        public bool valid;
        public string parseError;

        public static AgentResponse Parse(string rawJson)
        {
            var resp = new AgentResponse();
            if (string.IsNullOrWhiteSpace(rawJson))
            {
                resp.parseError = "empty response";
                return resp;
            }

            // Strip thinking tags (Qwen3.5 thinking mode safety)
            rawJson = StripThinking(rawJson);

            // Extract JSON block (LLM may wrap in markdown ```json fences)
            string json = ExtractJson(rawJson);
            if (string.IsNullOrEmpty(json))
            {
                resp.parseError = "no JSON found";
                resp.dialogue = rawJson.Trim();   // fallback: treat whole reply as plain dialogue
                return resp;
            }

            try
            {
                // Use SimpleJSON-style minimal parse (Unity's JsonUtility cannot do nested dicts)
                var parsed = MiniJson.Deserialize(json) as Dictionary<string, object>;
                if (parsed == null) { resp.parseError = "not a JSON object"; return resp; }

                if (parsed.TryGetValue("dialogue", out var dlg))
                    resp.dialogue = dlg?.ToString() ?? "";

                if (parsed.TryGetValue("tool_calls", out var calls) && calls is List<object> list)
                {
                    foreach (var c in list)
                    {
                        if (c is Dictionary<string, object> cd)
                        {
                            var tc = new ToolCall();
                            if (cd.TryGetValue("name", out var n)) tc.name = n?.ToString();
                            if (cd.TryGetValue("args", out var a) && a is Dictionary<string, object> ad)
                            {
                                foreach (var kv in ad) tc.args[kv.Key] = kv.Value;
                            }
                            if (!string.IsNullOrEmpty(tc.name)) resp.toolCalls.Add(tc);
                        }
                    }
                }
                resp.valid = true;
            }
            catch (Exception ex)
            {
                resp.parseError = $"parse exception: {ex.Message}";
                resp.dialogue = rawJson.Trim();
            }
            return resp;
        }

        static string StripThinking(string s)
        {
            // Remove <think>...</think> blocks
            int start = s.IndexOf("<think>", StringComparison.Ordinal);
            while (start != -1)
            {
                int end = s.IndexOf("</think>", start, StringComparison.Ordinal);
                if (end == -1) { s = s.Substring(0, start); break; }
                s = s.Remove(start, end - start + "</think>".Length);
                start = s.IndexOf("<think>", StringComparison.Ordinal);
            }
            return s;
        }

        static string ExtractJson(string s)
        {
            // Look for first '{' and matching '}'
            int start = s.IndexOf('{');
            if (start == -1) return null;
            int depth = 0;
            bool inString = false;
            bool escape = false;
            for (int i = start; i < s.Length; i++)
            {
                char c = s[i];
                if (escape) { escape = false; continue; }
                if (c == '\\') { escape = true; continue; }
                if (c == '"') { inString = !inString; continue; }
                if (inString) continue;
                if (c == '{') depth++;
                else if (c == '}') { depth--; if (depth == 0) return s.Substring(start, i - start + 1); }
            }
            return null;
        }
    }

    /// <summary>
    /// Minimal JSON parser (Unity's JsonUtility can't handle nested Dictionary).
    /// Adapted from public-domain MiniJSON.
    /// </summary>
    public static class MiniJson
    {
        public static object Deserialize(string json)
        {
            if (json == null) return null;
            return Parser.Parse(json);
        }

        sealed class Parser : IDisposable
        {
            const string WORD_BREAK = "{}[],:\"";
            public static bool IsWordBreak(char c) => char.IsWhiteSpace(c) || WORD_BREAK.IndexOf(c) != -1;

            enum TOKEN { NONE, CURLY_OPEN, CURLY_CLOSE, SQUARED_OPEN, SQUARED_CLOSE, COLON, COMMA, STRING, NUMBER, TRUE, FALSE, NULL }

            System.IO.StringReader json;
            Parser(string jsonString) { json = new System.IO.StringReader(jsonString); }
            public static object Parse(string jsonString) { using (var p = new Parser(jsonString)) return p.ParseValue(); }
            public void Dispose() { json.Dispose(); json = null; }

            Dictionary<string, object> ParseObject()
            {
                var table = new Dictionary<string, object>();
                json.Read(); // {
                while (true)
                {
                    switch (NextToken)
                    {
                        case TOKEN.NONE: return null;
                        case TOKEN.COMMA: continue;
                        case TOKEN.CURLY_CLOSE: return table;
                        default:
                            string name = ParseString();
                            if (name == null) return null;
                            if (NextToken != TOKEN.COLON) return null;
                            json.Read();
                            table[name] = ParseValue();
                            break;
                    }
                }
            }

            List<object> ParseArray()
            {
                var array = new List<object>();
                json.Read();
                bool parsing = true;
                while (parsing)
                {
                    TOKEN nextToken = NextToken;
                    switch (nextToken)
                    {
                        case TOKEN.NONE: return null;
                        case TOKEN.COMMA: continue;
                        case TOKEN.SQUARED_CLOSE: parsing = false; break;
                        default: array.Add(ParseByToken(nextToken)); break;
                    }
                }
                return array;
            }

            object ParseValue() => ParseByToken(NextToken);

            object ParseByToken(TOKEN token)
            {
                switch (token)
                {
                    case TOKEN.STRING: return ParseString();
                    case TOKEN.NUMBER: return ParseNumber();
                    case TOKEN.CURLY_OPEN: return ParseObject();
                    case TOKEN.SQUARED_OPEN: return ParseArray();
                    case TOKEN.TRUE: return true;
                    case TOKEN.FALSE: return false;
                    case TOKEN.NULL: return null;
                    default: return null;
                }
            }

            string ParseString()
            {
                var s = new System.Text.StringBuilder();
                char c;
                json.Read(); // skip "
                bool parsing = true;
                while (parsing)
                {
                    if (json.Peek() == -1) break;
                    c = NextChar;
                    if (c == '"') { parsing = false; break; }
                    if (c == '\\')
                    {
                        if (json.Peek() == -1) break;
                        c = NextChar;
                        switch (c)
                        {
                            case '"': case '\\': case '/': s.Append(c); break;
                            case 'b': s.Append('\b'); break;
                            case 'f': s.Append('\f'); break;
                            case 'n': s.Append('\n'); break;
                            case 'r': s.Append('\r'); break;
                            case 't': s.Append('\t'); break;
                            case 'u':
                                var hex = new char[4];
                                for (int i = 0; i < 4; i++) hex[i] = NextChar;
                                s.Append((char)Convert.ToInt32(new string(hex), 16));
                                break;
                        }
                    }
                    else s.Append(c);
                }
                return s.ToString();
            }

            object ParseNumber()
            {
                string number = NextWord;
                if (number.IndexOf('.') == -1 && number.IndexOf('e') == -1 && number.IndexOf('E') == -1)
                {
                    long.TryParse(number, out long parsedInt);
                    return parsedInt;
                }
                double.TryParse(number, System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out double parsedDouble);
                return parsedDouble;
            }

            void EatWhitespace() { while (char.IsWhiteSpace((char)json.Peek())) { json.Read(); if (json.Peek() == -1) break; } }
            char NextChar => Convert.ToChar(json.Read());
            string NextWord
            {
                get
                {
                    var word = new System.Text.StringBuilder();
                    while (!IsWordBreak((char)json.Peek())) { word.Append(NextChar); if (json.Peek() == -1) break; }
                    return word.ToString();
                }
            }
            TOKEN NextToken
            {
                get
                {
                    EatWhitespace();
                    if (json.Peek() == -1) return TOKEN.NONE;
                    char c = (char)json.Peek();
                    switch (c)
                    {
                        case '{': return TOKEN.CURLY_OPEN;
                        case '}': json.Read(); return TOKEN.CURLY_CLOSE;
                        case '[': return TOKEN.SQUARED_OPEN;
                        case ']': json.Read(); return TOKEN.SQUARED_CLOSE;
                        case ',': json.Read(); return TOKEN.COMMA;
                        case '"': return TOKEN.STRING;
                        case ':': return TOKEN.COLON;
                        case '0': case '1': case '2': case '3': case '4':
                        case '5': case '6': case '7': case '8': case '9':
                        case '-': return TOKEN.NUMBER;
                    }
                    string word = NextWord;
                    switch (word)
                    {
                        case "false": return TOKEN.FALSE;
                        case "true": return TOKEN.TRUE;
                        case "null": return TOKEN.NULL;
                    }
                    return TOKEN.NONE;
                }
            }
        }
    }
}
