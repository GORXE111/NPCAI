using System;
using System.Collections.Generic;
using UnityEngine;

namespace NPCLLM.NPC
{
    public enum NPCState { Idle, Working, Walking, Chatting, Trading }

    [Serializable]
    public class ScheduleEntry
    {
        public TimePeriod period;
        public NPCState action;
    }

    /// <summary>
    /// NPC behavior state machine. Drives movement, idle actions, and schedule transitions.
    /// Works alongside NPCBrain (dialogue) and TradeInventory (trading).
    /// </summary>
    public class NPCBehavior : MonoBehaviour
    {
        [Header("State")]
        [SerializeField] private NPCState currentState = NPCState.Idle;
        [SerializeField] private string currentAction = "standing";

        [Header("Movement")]
        [SerializeField] private float moveSpeed = 2.5f;
        [SerializeField] private float arrivalThreshold = 0.2f;

        [Header("Schedule")]
        [SerializeField] private List<ScheduleEntry> schedule = new List<ScheduleEntry>();

        // Idle behavior
        private float _idleTimer;
        private float _idleDuration;
        private const float IDLE_MIN = 5f;
        private const float IDLE_MAX = 15f;

        // Walking
        private Vector3 _targetPosition;
        private NPCState _stateAfterArrival = NPCState.Idle;

        // Chatting
        private NPCState _stateBeforeChat;

        // References
        private Vector3 _homePosition;
        private string _npcName;
        private string _workAction = "working";

        // Idle action descriptions per occupation
        private static readonly Dictionary<string, string> WorkActions = new Dictionary<string, string>
        {
            { "blacksmith", "hammering at the anvil" },
            { "herbalist", "sorting dried herbs" },
            { "merchant", "counting coins" },
            { "guard captain", "surveying the crowd" },
            { "innkeeper", "wiping the counter" },
            { "hunter", "sharpening arrows" },
            { "priestess", "reading scripture" },
            { "pickpocket", "leaning against the wall" },
            { "fisherman", "mending nets" },
            { "baker", "kneading dough" }
        };

        public NPCState CurrentState => currentState;
        public string CurrentAction => currentAction;
        public Vector3 HomePosition => _homePosition;

        public void Initialize(string npcName, string occupation, Vector3 homePosition)
        {
            _npcName = npcName;
            _homePosition = homePosition;
            transform.position = homePosition;

            if (WorkActions.ContainsKey(occupation))
                _workAction = WorkActions[occupation];

            // Default schedule
            if (schedule.Count == 0)
            {
                schedule.Add(new ScheduleEntry { period = TimePeriod.Dawn, action = NPCState.Walking });
                schedule.Add(new ScheduleEntry { period = TimePeriod.Morning, action = NPCState.Working });
                schedule.Add(new ScheduleEntry { period = TimePeriod.Afternoon, action = NPCState.Working });
                schedule.Add(new ScheduleEntry { period = TimePeriod.Evening, action = NPCState.Idle });
                schedule.Add(new ScheduleEntry { period = TimePeriod.Night, action = NPCState.Idle });
            }

            // Subscribe to day clock
            if (DayClock.Instance != null)
                DayClock.Instance.OnPeriodChanged += OnPeriodChanged;

            // Start in appropriate state
            if (DayClock.Instance != null)
                OnPeriodChanged(DayClock.Instance.CurrentPeriod);
            else
                EnterState(NPCState.Working);
        }

        private void OnDestroy()
        {
            if (DayClock.Instance != null)
                DayClock.Instance.OnPeriodChanged -= OnPeriodChanged;
        }

        private void Update()
        {
            UpdateState();
        }

        // ── State Machine ──────────────────────────

        public void EnterState(NPCState newState)
        {
            if (currentState == NPCState.Chatting && newState != NPCState.Chatting)
            {
                // Chatting exit is managed by EndChat()
            }

            currentState = newState;

            switch (newState)
            {
                case NPCState.Idle:
                    _idleDuration = UnityEngine.Random.Range(IDLE_MIN, IDLE_MAX);
                    _idleTimer = 0f;
                    currentAction = "standing around";
                    break;

                case NPCState.Working:
                    currentAction = _workAction;
                    // Walk home if not there
                    if (Vector3.Distance(transform.position, _homePosition) > 1f)
                    {
                        WalkTo(_homePosition, NPCState.Working);
                        return;
                    }
                    break;

                case NPCState.Walking:
                    currentAction = "walking";
                    break;

                case NPCState.Chatting:
                    currentAction = "chatting";
                    break;

                case NPCState.Trading:
                    currentAction = "trading";
                    break;
            }
        }

        private void UpdateState()
        {
            switch (currentState)
            {
                case NPCState.Idle:
                    UpdateIdle();
                    break;
                case NPCState.Working:
                    // Just stay at stall, no movement
                    break;
                case NPCState.Walking:
                    UpdateWalking();
                    break;
                case NPCState.Chatting:
                    // Locked until EndChat() is called
                    break;
                case NPCState.Trading:
                    // Locked until trade ends
                    break;
            }
        }

        private void UpdateIdle()
        {
            _idleTimer += Time.deltaTime;
            if (_idleTimer >= _idleDuration)
            {
                // Random: 60% wander, 40% go back to stall
                if (UnityEngine.Random.value < 0.6f)
                {
                    Vector3 wanderTarget = NPCLocations.GetRandomWanderPoint();
                    WalkTo(wanderTarget, NPCState.Idle);
                }
                else
                {
                    WalkTo(_homePosition, NPCState.Working);
                }
            }
        }

        private void UpdateWalking()
        {
            Vector3 current = transform.position;
            Vector3 target = new Vector3(_targetPosition.x, current.y, _targetPosition.z);

            transform.position = Vector3.MoveTowards(current, target, moveSpeed * Time.deltaTime);

            if (Vector3.Distance(transform.position, target) < arrivalThreshold)
            {
                EnterState(_stateAfterArrival);
            }
        }

        // ── Public API ─────────────────────────────

        public void WalkTo(Vector3 target, NPCState onArrival)
        {
            _targetPosition = target;
            _stateAfterArrival = onArrival;
            currentState = NPCState.Walking;
            currentAction = "walking";
        }

        /// <summary>
        /// Called by SocialDirector when an NPC-NPC chat begins.
        /// </summary>
        public void StartChat(NPCBehavior otherNpc)
        {
            _stateBeforeChat = currentState;
            EnterState(NPCState.Chatting);

            // Face the other NPC (rotate on Y only)
            Vector3 dir = otherNpc.transform.position - transform.position;
            if (dir.sqrMagnitude > 0.01f)
            {
                dir.y = 0;
                transform.forward = dir.normalized;
            }
        }

        /// <summary>
        /// Called when NPC-NPC chat ends.
        /// </summary>
        public void EndChat()
        {
            EnterState(_stateBeforeChat);
        }

        /// <summary>
        /// Called when player initiates trade.
        /// </summary>
        public void StartTrade()
        {
            _stateBeforeChat = currentState;
            EnterState(NPCState.Trading);
        }

        public void EndTrade()
        {
            EnterState(_stateBeforeChat);
        }

        public bool IsAvailableForChat()
        {
            return currentState == NPCState.Idle || currentState == NPCState.Working;
        }

        // ── Schedule ───────────────────────────────

        private void OnPeriodChanged(TimePeriod period)
        {
            // Don't interrupt chatting or trading
            if (currentState == NPCState.Chatting || currentState == NPCState.Trading)
                return;

            var entry = schedule.Find(s => s.period == period);
            if (entry == null) return;

            switch (entry.action)
            {
                case NPCState.Working:
                    if (Vector3.Distance(transform.position, _homePosition) > 1f)
                        WalkTo(_homePosition, NPCState.Working);
                    else
                        EnterState(NPCState.Working);
                    break;

                case NPCState.Idle:
                    EnterState(NPCState.Idle);
                    break;

                case NPCState.Walking:
                    // Dawn: walk to stall
                    WalkTo(_homePosition, NPCState.Working);
                    break;
            }
        }
    }
}
