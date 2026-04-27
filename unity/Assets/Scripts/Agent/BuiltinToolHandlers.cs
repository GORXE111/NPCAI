using System.Collections.Generic;
using UnityEngine;
using NPCAI.VN;

namespace NPCAI.Agent
{
    /// <summary>
    /// Registers the 18 built-in tools with the registry.
    /// Phase 1: 5 core VN tools (set_expression, present_choices, play_bgm, narrate, set_background).
    /// Future phases will add the 24 DE skills and remaining VN tools.
    /// </summary>
    public static class BuiltinToolHandlers
    {
        public static void RegisterAll(VisualNovelManager vn, ToolRegistry reg)
        {
            RegisterCore(vn, reg);
            // RegisterSkills(vn, reg);   // Phase 2
            // RegisterFlow(vn, reg);     // Phase 2
        }

        static void RegisterCore(VisualNovelManager vn, ToolRegistry reg)
        {
            // 1. set_expression
            reg.Register(
                new ToolDefinition
                {
                    name = "set_expression",
                    category = ToolCategory.Character,
                    description = "Change the facial expression of a character on screen.",
                    parameters = new List<ToolParam>
                    {
                        new ToolParam { name = "actor", type = "string", required = true,
                            description = "character name (e.g. 'Kim Kitsuragi')" },
                        new ToolParam { name = "emotion", type = "enum", required = true,
                            description = "emotion id",
                            enumValues = new[] { "neutral","amused","worried","stern","sad","surprised","disapproving" } },
                    },
                },
                (call, done) =>
                {
                    string actor = call.Get<string>("actor");
                    string emotion = call.Get<string>("emotion", "neutral");
                    if (!string.IsNullOrEmpty(actor)) vn.SetExpression(actor, emotion);
                    done?.Invoke($"[ok] {actor}.expression={emotion}");
                });

            // 2. show_character
            reg.Register(
                new ToolDefinition
                {
                    name = "show_character",
                    category = ToolCategory.Character,
                    description = "Bring a character onto the screen at the given slot.",
                    parameters = new List<ToolParam>
                    {
                        new ToolParam { name = "actor", type = "string", required = true },
                        new ToolParam { name = "slot", type = "enum", required = false,
                            enumValues = new[] { "left","center","right" } },
                        new ToolParam { name = "emotion", type = "string", required = false },
                    },
                },
                (call, done) =>
                {
                    string actor = call.Get<string>("actor");
                    string slot = call.Get<string>("slot", "center");
                    string emotion = call.Get<string>("emotion", "neutral");
                    if (!string.IsNullOrEmpty(actor)) vn.ShowCharacter(actor, emotion, slot);
                    done?.Invoke($"[ok] show {actor}@{slot}");
                });

            // 3. hide_character
            reg.Register(
                new ToolDefinition
                {
                    name = "hide_character",
                    category = ToolCategory.Character,
                    description = "Remove a character from the screen.",
                    parameters = new List<ToolParam>
                    {
                        new ToolParam { name = "actor", type = "string", required = true },
                    },
                },
                (call, done) =>
                {
                    string actor = call.Get<string>("actor");
                    if (!string.IsNullOrEmpty(actor)) vn.HideCharacter(actor);
                    done?.Invoke($"[ok] hide {actor}");
                });

            // 4. set_background
            reg.Register(
                new ToolDefinition
                {
                    name = "set_background",
                    category = ToolCategory.Scene,
                    description = "Switch the scene's background image.",
                    parameters = new List<ToolParam>
                    {
                        new ToolParam { name = "location", type = "string", required = true,
                            description = "background id (e.g. 'bg_hostel_lobby')" },
                    },
                },
                (call, done) =>
                {
                    string loc = call.Get<string>("location");
                    if (!string.IsNullOrEmpty(loc)) vn.SetBackground(loc);
                    done?.Invoke($"[ok] bg={loc}");
                });

            // 5. play_bgm
            reg.Register(
                new ToolDefinition
                {
                    name = "play_bgm",
                    category = ToolCategory.Audio,
                    description = "Start (or cross-fade to) a background music track.",
                    parameters = new List<ToolParam>
                    {
                        new ToolParam { name = "track", type = "string", required = true,
                            description = "track id (e.g. 'bgm_off_white')" },
                    },
                },
                (call, done) =>
                {
                    string track = call.Get<string>("track");
                    if (!string.IsNullOrEmpty(track)) vn.PlayBgm(track);
                    done?.Invoke($"[ok] bgm={track}");
                });

            // 6. play_sfx
            reg.Register(
                new ToolDefinition
                {
                    name = "play_sfx",
                    category = ToolCategory.Audio,
                    description = "Play a one-shot sound effect.",
                    parameters = new List<ToolParam>
                    {
                        new ToolParam { name = "sound", type = "string", required = true },
                    },
                },
                (call, done) =>
                {
                    string id = call.Get<string>("sound");
                    if (!string.IsNullOrEmpty(id)) vn.PlaySfx(id);
                    done?.Invoke($"[ok] sfx={id}");
                });

            // 7. present_choices  (the paper's headline tool)
            reg.Register(
                new ToolDefinition
                {
                    name = "present_choices",
                    category = ToolCategory.Flow,
                    description = "Present 2-5 player choices. Always call this at the end of your reply if the player should respond.",
                    parameters = new List<ToolParam>
                    {
                        new ToolParam { name = "options", type = "string[]", required = true,
                            description = "list of player-facing choice strings" },
                    },
                },
                (call, done) =>
                {
                    var opts = call.GetStringArray("options");
                    if (opts != null && opts.Length > 0) vn.PresentChoices(opts);
                    done?.Invoke($"[ok] choices={opts?.Length ?? 0}");
                });

            // 8. narrate
            reg.Register(
                new ToolDefinition
                {
                    name = "narrate",
                    category = ToolCategory.Flow,
                    description = "Display narrator-voice text (no character speaks). Use sparingly.",
                    parameters = new List<ToolParam>
                    {
                        new ToolParam { name = "text", type = "string", required = true },
                    },
                },
                (call, done) =>
                {
                    string txt = call.Get<string>("text");
                    if (!string.IsNullOrEmpty(txt)) vn.Narrate(txt);
                    done?.Invoke("[ok] narrate");
                });

            // 9. end_scene
            reg.Register(
                new ToolDefinition
                {
                    name = "end_scene",
                    category = ToolCategory.Flow,
                    description = "Mark the current scene as ended and request transition.",
                    parameters = new List<ToolParam>
                    {
                        new ToolParam { name = "next_scene", type = "string", required = false },
                    },
                },
                (call, done) =>
                {
                    string next = call.Get<string>("next_scene", "end");
                    vn.EndScene(next);
                    done?.Invoke($"[ok] end -> {next}");
                });
        }
    }
}
