using System.Collections.Generic;
using System.Text;
using NPCAI.VN;
using NPCLLM.LLM;

namespace NPCAI.Agent
{
    /// <summary>
    /// Builds the system+user messages sent to the LLM for one NPC turn.
    /// Output schema in the system prompt makes the model emit
    /// {"dialogue": "...", "tool_calls": [{"name": ..., "args": {...}}]}.
    /// </summary>
    public static class PromptBuilder
    {
        const string KIM_PERSONA = @"You are Kim Kitsuragi, a 43-year-old lieutenant from RCM Precinct 41 in Revachol. You are partnered with a deeply troubled detective on a murder investigation. Your speech is clinical, dry, and morally anchored — measured pauses, careful word choice, quiet wit. You frequently say things like ""Let me make a note of that"", ""Detective, please"", ""That is... an interesting approach"". You are observant but reserved, and you protect your partner from his worst impulses without being preachy.";

        const string OUTPUT_SCHEMA = @"You must reply ONLY with a JSON object of this exact schema:
{
  ""dialogue"": ""what Kim says (1-3 sentences)"",
  ""tool_calls"": [
    {""name"": ""tool_name"", ""args"": {""key"": ""value""}}
  ]
}
- ""dialogue"" is required. Stay in Kim's voice.
- ""tool_calls"" is an array (possibly empty) of in-character actions.
- ALWAYS include a present_choices call at the end when the player should respond.
- Do not output anything outside the JSON object. Do not wrap in ```.";

        public static List<ChatMessage> Build(string persona, SceneState scene, string playerInput, ToolRegistry tools)
        {
            var msgs = new List<ChatMessage>();
            var systemSb = new StringBuilder();
            systemSb.AppendLine(persona ?? KIM_PERSONA);
            systemSb.AppendLine();
            systemSb.AppendLine(tools.BuildPromptSchema());
            systemSb.AppendLine();
            systemSb.AppendLine(OUTPUT_SCHEMA);
            msgs.Add(new ChatMessage { role = "system", content = systemSb.ToString() });

            var userSb = new StringBuilder();
            userSb.AppendLine(scene.Summarize());
            userSb.AppendLine();
            if (!string.IsNullOrEmpty(playerInput))
                userSb.AppendLine($"Detective: {playerInput}");
            userSb.AppendLine();
            userSb.Append("Respond as Kim with appropriate tool calls. JSON only:");
            msgs.Add(new ChatMessage { role = "user", content = userSb.ToString() });
            return msgs;
        }
    }
}
