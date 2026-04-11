using System;
using System.Globalization;
using System.IO;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace NPCLLM
{
    /// <summary>
    /// Test bridge - file communication mode.
    /// Python writes .test_command -> TestBridge executes -> writes .test_result
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class TestBridge : MonoBehaviour
    {
        [Serializable]
        private struct TestResponse
        {
            public bool success;
            public string command;
            public string route;
            public string action;
            public string target;
            public string message;
            public string value;
            public float number;
        }

        public static TestBridge Instance { get; private set; }

        [SerializeField] private string pendingCommand;
        [SerializeField] private string commandResult;
        [SerializeField] private bool isBusy;

        private string _cmdFile;
        private string _resultFile;
        private string _logFile;

        // For async NPC responses
        private bool _waitingForNPC;
        private string _waitingCommand;

        private void Awake()
        {
            if (Instance != null && Instance != this) { Destroy(gameObject); return; }
            Instance = this;
            DontDestroyOnLoad(gameObject);

            string projectRoot = Path.GetDirectoryName(Application.dataPath);
            _cmdFile = Path.Combine(projectRoot, ".test_command");
            _resultFile = Path.Combine(projectRoot, ".test_result");
            _logFile = Path.Combine(projectRoot, ".test_log");

            TryDelete(_cmdFile);
            TryDelete(_resultFile);
        }

        private void Update()
        {
            if (Time.timeScale < 0.01f) Time.timeScale = 1f;

            if (_waitingForNPC) return;

            // File commands
            if (!isBusy && File.Exists(_cmdFile))
            {
                try
                {
                    string cmd = File.ReadAllText(_cmdFile).Trim();
                    File.Delete(_cmdFile);
                    TryDelete(_resultFile); // Clear old result before processing new command
                    if (!string.IsNullOrEmpty(cmd))
                    {
                        // NPC talk is async
                        if (cmd.StartsWith("npc:talk:", StringComparison.OrdinalIgnoreCase))
                        {
                            HandleNpcTalkAsync(cmd);
                            return;
                        }

                        string result = ExecuteCommand(cmd);
                        WriteResult(result);
                    }
                }
                catch (Exception ex)
                {
                    Debug.LogError("[TestBridge] File cmd error: " + ex.Message);
                }
                return;
            }

            // MCP mode (SerializeField)
            if (!isBusy && !string.IsNullOrWhiteSpace(pendingCommand))
            {
                string command = pendingCommand.Trim();
                pendingCommand = string.Empty;

                if (command.StartsWith("npc:talk:", StringComparison.OrdinalIgnoreCase))
                {
                    HandleNpcTalkAsync(command);
                    return;
                }

                string result = ExecuteCommand(command);
                commandResult = result;
                Debug.Log("[TEST_RESULT]" + result);
            }
        }

        private void WriteResult(string result)
        {
            File.WriteAllText(_resultFile, result);
            File.AppendAllText(_logFile, result + "\n");
            Debug.Log("[TEST_RESULT]" + result);
        }

        private void HandleNpcTalkAsync(string command)
        {
            // Format: npc:talk:NpcName:Message
            string[] parts = command.Split(new char[] { ':' }, 4);
            if (parts.Length < 4)
            {
                WriteResult(ToJson(Err(command, "npc", "talk", "Format: npc:talk:Name:Message")));
                return;
            }

            string npcName = parts[2].Trim();
            string message = parts[3].Trim();

            var townManager = FindFirstObjectByType<TownManager>();
            if (townManager == null)
            {
                WriteResult(ToJson(Err(command, "npc", "talk", "TownManager not found")));
                return;
            }

            _waitingForNPC = true;
            _waitingCommand = command;
            isBusy = true;

            townManager.TalkToNPC(npcName, message, response =>
            {
                var resp = new TestResponse
                {
                    success = true,
                    command = command,
                    route = "npc",
                    action = "talk",
                    target = npcName,
                    message = response,
                    value = message,
                    number = 1f
                };
                string result = ToJson(resp);
                commandResult = result;
                WriteResult(result);
                _waitingForNPC = false;
                isBusy = false;
            });
        }

        public string ExecuteCommand(string command)
        {
            if (string.IsNullOrWhiteSpace(command))
                return ToJson(Err(command, "system", "empty", "Command is empty."));

            string[] segments = command.Split(':');
            string route = segments[0].Trim().ToLowerInvariant();

            try
            {
                switch (route)
                {
                    case "query": return ToJson(HandleQuery(command, segments));
                    case "screenshot": return ToJson(HandleScreenshot(command, segments));
                    case "npc": return ToJson(HandleNpcCommand(command, segments));
                    case "propagate": return ToJson(HandlePropagation(command, segments));
                    default: return ToJson(Err(command, route, "unknown", "Unknown route."));
                }
            }
            catch (Exception ex)
            {
                return ToJson(Err(command, route, "exception", ex.Message));
            }
        }

        private TestResponse HandleQuery(string command, string[] segments)
        {
            string action = Arg(segments, 1, "scene").ToLowerInvariant();
            string target = Arg(segments, 2, "");

            switch (action)
            {
                case "exists":
                {
                    var go = GameObject.Find(target);
                    return Ok(command, "query", action, target, go != null ? "found" : "not found", go != null ? 1f : 0f);
                }
                case "npcs":
                {
                    var tm = FindFirstObjectByType<TownManager>();
                    if (tm == null) return Err(command, "query", "npcs", "TownManager not found");
                    var npcs = tm.GetAllNPCs();
                    string names = "";
                    foreach (var npc in npcs)
                        names += npc.NpcName + " (" + npc.Personality.occupation + "), ";
                    return Ok(command, "query", "npcs", "all", names.TrimEnd(',', ' '), npcs.Count);
                }
                case "scene":
                default:
                {
                    var scene = SceneManager.GetActiveScene();
                    return Ok(command, "query", "scene", scene.name, scene.path, scene.rootCount);
                }
            }
        }

        private TestResponse HandleScreenshot(string command, string[] segments)
        {
            string fileName = Arg(segments, 1, "");
            if (string.IsNullOrWhiteSpace(fileName))
                fileName = string.Format(CultureInfo.InvariantCulture, "test_{0:yyyyMMdd_HHmmss}", DateTime.Now);
            string path = Path.Combine(Application.persistentDataPath, fileName + ".png");
            ScreenCapture.CaptureScreenshot(path);
            return Ok(command, "screenshot", "capture", fileName, path, 0f);
        }

        private TestResponse HandleNpcCommand(string command, string[] segments)
        {
            string action = Arg(segments, 1, "list").ToLowerInvariant();

            switch (action)
            {
                case "list":
                {
                    var tm = FindFirstObjectByType<TownManager>();
                    if (tm == null) return Err(command, "npc", "list", "TownManager not found");
                    var npcs = tm.GetAllNPCs();
                    string names = "";
                    foreach (var npc in npcs)
                        names += npc.NpcName + ",";
                    return Ok(command, "npc", "list", "all", names.TrimEnd(','), npcs.Count);
                }
                case "memory":
                {
                    string npcName = Arg(segments, 2, "");
                    var tm = FindFirstObjectByType<TownManager>();
                    if (tm == null) return Err(command, "npc", "memory", "TownManager not found");
                    var npcs = tm.GetAllNPCs();
                    var target = npcs.Find(n => n.NpcName == npcName);
                    if (target == null) return Err(command, "npc", "memory", "NPC not found: " + npcName);
                    string summary = target.Memory.GetMemorySummary(10);
                    return Ok(command, "npc", "memory", npcName, summary, target.Memory.MemoryCount);
                }
                default:
                    return Err(command, "npc", action, "Unknown npc action. Use: list, talk, memory");
            }
        }

        private TestResponse HandlePropagation(string command, string[] segments)
        {
            // Format: propagate:NpcName:information
            string npcName = Arg(segments, 1, "");
            string info = Arg(segments, 2, "");
            if (string.IsNullOrEmpty(npcName) || string.IsNullOrEmpty(info))
                return Err(command, "propagate", "start", "Format: propagate:NpcName:information");

            var tm = FindFirstObjectByType<TownManager>();
            if (tm == null) return Err(command, "propagate", "start", "TownManager not found");

            tm.StartPropagationExperiment(npcName, info);
            return Ok(command, "propagate", "start", npcName, "Propagation started: " + info, 1f);
        }

        private static string Arg(string[] s, int i, string d) =>
            i < s.Length && !string.IsNullOrWhiteSpace(s[i]) ? s[i].Trim() : d;
        private static string ToJson(TestResponse r) => JsonUtility.ToJson(r);
        private static TestResponse Ok(string cmd, string route, string action, string target, string msg, float num) =>
            new TestResponse { success = true, command = cmd, route = route, action = action, target = target, message = msg, value = "", number = num };
        private static TestResponse Err(string cmd, string route, string action, string msg) =>
            new TestResponse { success = false, command = cmd, route = route, action = action, target = "", message = msg, value = "", number = -1f };
        private static void TryDelete(string path)
        {
            try { if (File.Exists(path)) File.Delete(path); } catch { }
        }
    }
}
