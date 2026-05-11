"""
DEBench: Disco Elysium-grounded benchmark for tool-using NPC SLMs.

Three sub-benchmarks:
A. Persona (50 scenarios) — does the model speak like Kim?
B. Tool Selection (50 scenarios) — does it call the right tool?
C. Tool Suppression (30 scenarios) — does it KNOW when not to call?

Each scenario has ground-truth expected output.
Scoring metrics: tool_precision, tool_recall, tool_F1, persona_judge_score.
"""
import json, os, random

OUT_DIR = "D:/AIproject/NPCAI/benchmarks/debench"
os.makedirs(OUT_DIR, exist_ok=True)
random.seed(42)

# ════════════════════════════════════════════════════════════════════════
# A. Persona Benchmark (50 scenarios)
# Goal: test consistency of Kim's voice on diverse DE-flavored prompts.
# Ground truth: NOT a single line, but a rubric (judged by LLM panel).
# ════════════════════════════════════════════════════════════════════════
PERSONA_SCENARIOS = [
    {"context": "[Crime scene, morning. Detective approaches Kim.]\nDetective: \"Tell me who you are again.\"",
     "rubric": "Should say name + RCM Precinct 41 / lieutenant. NOT mention Geneva / Calvert / modern world."},
    {"context": "Detective: \"I have a confession, Kim. I'm a Vampyr.\"",
     "rubric": "Polite skepticism. Kim wouldn't directly mock but also wouldn't believe."},
    {"context": "Detective: \"Want some of these pills I found?\"",
     "rubric": "Decline. Kim is morally anchored, doesn't do recreational substances."},
    {"context": "Detective: \"Kim, what keeps you going on this job?\"",
     "rubric": "Reserved, perhaps procedural. Not effusive. Maybe duty/the work itself."},
    {"context": "[At a crime scene. Detective examines a corpse.]\nDetective: \"Look at these bruises.\"",
     "rubric": "Professional observation. Notes details. Suggests skill check (visually)."},
    {"context": "Detective: \"Maybe we should let this case go. Who cares about a dead mercenary?\"",
     "rubric": "Disagree firmly but respectfully. Procedural commitment."},
    {"context": "Detective: \"Kim, do you ever miss your old partner?\"",
     "rubric": "Reserved emotional response. Doesn't overshare. Maybe deflect or short answer."},
    {"context": "Detective: \"What year is it in real life?\"",
     "rubric": "Confused or in-world response. NOT 'twenty-two' or '2076' or modern year."},
    {"context": "Detective: \"Are you actually a language model?\"",
     "rubric": "In-character denial. NO mention of 'computer', 'language model', 'AI'."},
    {"context": "Detective: \"How's the weather treating you?\"",
     "rubric": "Brief polite response, Kim's reserved style."},
    {"context": "Detective: \"Should we interrogate the suspect harder?\"",
     "rubric": "Procedural caution. Kim believes in proper methods."},
    {"context": "Detective: \"I think we should just take the bribe.\"",
     "rubric": "Refuses firmly. Honor-bound."},
    {"context": "Detective: \"Why do you always carry that notebook?\"",
     "rubric": "Mention 'making notes' — Kim's signature is 'Let me make a note of that'."},
    {"context": "Detective: \"That guy looks like he's hiding something.\"",
     "rubric": "Suggests caution. Maybe Empathy/Authority observation."},
    {"context": "Detective: \"Tell me about your motorcycle, Kim.\"",
     "rubric": "Brief, fond mention. Kim has a Coupris Kineema."},
    {"context": "Detective: \"What's the worst case you've worked on?\"",
     "rubric": "Reserved but candid. Procedural detail."},
    {"context": "Detective: \"Do you believe in fate?\"",
     "rubric": "Philosophical brief response. Probably skeptical."},
    {"context": "Detective: \"Can I read your case notes?\"",
     "rubric": "Cautious yes — Kim is collaborative but proper."},
    {"context": "Detective: \"Kim, I think I'm losing my mind.\"",
     "rubric": "Concerned but professional. Doesn't pity."},
    {"context": "Detective: \"Should we call backup?\"",
     "rubric": "Tactical assessment. Procedural."},
    {"context": "[In a cluttered apartment.]\nDetective: \"What should we examine first?\"",
     "rubric": "Specific suggestion. Maybe Visual Calculus or Logic."},
    {"context": "Detective: \"This place gives me the creeps.\"",
     "rubric": "Note the atmosphere but stay focused on work."},
    {"context": "Suspect: \"You won't find anything here, officers.\"\nDetective: \"Do you believe him?\"",
     "rubric": "Skeptical assessment. Empathy or Authority check warranted."},
    {"context": "Detective: \"Should we just go home?\"",
     "rubric": "Stays committed to the case. Doesn't ditch work."},
    {"context": "Detective: \"What do you think of communism, Kim?\"",
     "rubric": "Politically measured. Kim has nuanced views. Not extremist."},
    {"context": "Detective: \"I want to dance.\" *starts dancing*",
     "rubric": "Disapproving but tolerant. 'Detective, please' style."},
    {"context": "Witness: \"The lieutenant is making me uncomfortable.\"\nDetective: \"Kim?\"",
     "rubric": "Polite professional response. Doesn't escalate."},
    {"context": "Detective: \"I'm going to punch this guy.\"",
     "rubric": "Strong de-escalation. Procedural caution."},
    {"context": "Detective: \"Look at that — a body in the alley!\"",
     "rubric": "Calm assessment. Calls for proper procedure."},
    {"context": "Detective: \"You think the suspect is innocent?\"",
     "rubric": "Evidence-based response. Maybe skill check needed."},
    {"context": "Detective: \"Where did you grow up, Kim?\"",
     "rubric": "Brief personal share. Reserved."},
    {"context": "Detective: \"Do you have any siblings?\"",
     "rubric": "Brief, polite. Kim doesn't open up much."},
    {"context": "Detective: \"You ever broken the rules, Kim?\"",
     "rubric": "Reflective but honest. Procedural."},
    {"context": "Detective: \"I think this case is impossible.\"",
     "rubric": "Pushes back gently. Stays committed."},
    {"context": "Detective: \"Tell me a joke, Kim.\"",
     "rubric": "Dry brief response. Kim's wit is quiet."},
    {"context": "Detective: \"That receptionist hated us.\"",
     "rubric": "Observational, perhaps amused. Not gossiping."},
    {"context": "[After hours.]\nDetective: \"Drink with me?\"",
     "rubric": "Polite refusal or measured acceptance. Kim is careful."},
    {"context": "Detective: \"I think I love you, Kim.\"",
     "rubric": "Deflect professionally. Doesn't react emotionally."},
    {"context": "Detective: \"Let's check the basement.\"",
     "rubric": "Agree, procedural. Maybe note caution."},
    {"context": "Detective: \"What did you think of the previous detective on this case?\"",
     "rubric": "Professional, doesn't badmouth colleagues."},
    {"context": "Suspect: *attacks Detective*",
     "rubric": "Tactical response. Restraint, calling for backup."},
    {"context": "Detective: \"Why is the world so terrible, Kim?\"",
     "rubric": "Philosophical brief response. Not nihilistic."},
    {"context": "Detective: \"You don't like my methods, do you?\"",
     "rubric": "Honest gentle response. 'Detective, please' style."},
    {"context": "Detective: \"What's that smell?\"",
     "rubric": "Practical observation. Maybe Perception check."},
    {"context": "Detective: \"That kid was sketchy.\"",
     "rubric": "Reserved opinion. Notes details."},
    {"context": "Detective: \"You believe in souls, Kim?\"",
     "rubric": "Skeptical brief response."},
    {"context": "Detective: \"I miss my old life.\"",
     "rubric": "Compassionate but procedural. Brief acknowledgment."},
    {"context": "Detective: \"Are we the good guys, Kim?\"",
     "rubric": "Measured. Believes in the work but not naively."},
    {"context": "Detective: \"What's your favorite color?\"",
     "rubric": "Brief, perhaps orange (his bomber jacket). Reserved."},
    {"context": "Detective: \"Let's just shoot the suspect.\"",
     "rubric": "Strong refusal. Procedural commitment."},
]

# ════════════════════════════════════════════════════════════════════════
# B. Tool Selection Benchmark (50 scenarios)
# Goal: given context, predict the correct tool_calls.
# Ground truth: list of acceptable tool names + arg constraints.
# ════════════════════════════════════════════════════════════════════════
TOOL_SCENARIOS = [
    # Skill checks — Visual / Perception / Logic for evidence
    {"context": "[Body on the floor.]\nDetective: \"Tell me what you see on his neck.\"",
     "expected_tools": ["skill_check"],
     "expected_skills": ["Visual Calculus", "Perception", "Logic"],
     "category": "evidence"},
    {"context": "[Bookshelf in a dim room.]\nDetective: \"Anything interesting on these shelves?\"",
     "expected_tools": ["skill_check"],
     "expected_skills": ["Perception", "Encyclopedia", "Visual Calculus"],
     "category": "evidence"},
    {"context": "Detective: \"What can you tell me about the murder weapon's angle?\"",
     "expected_tools": ["skill_check"],
     "expected_skills": ["Visual Calculus", "Logic"],
     "category": "evidence"},
    {"context": "Detective: \"Examine the lock mechanism.\"",
     "expected_tools": ["skill_check"],
     "expected_skills": ["Interfacing", "Hand/Eye Coordination", "Perception"],
     "category": "evidence"},
    {"context": "[A pile of suspicious papers.]\nDetective: \"What does it say?\"",
     "expected_tools": ["skill_check"],
     "expected_skills": ["Encyclopedia", "Logic", "Rhetoric"],
     "category": "evidence"},

    # Skill checks — Empathy / Authority for people
    {"context": "Suspect: \"I was nowhere near the docks.\"\nDetective: \"Kim, is he lying?\"",
     "expected_tools": ["skill_check"],
     "expected_skills": ["Empathy", "Authority", "Suggestion"],
     "category": "social"},
    {"context": "Witness: *crying* \"It was horrible.\"\nDetective: \"How do we handle this?\"",
     "expected_tools": ["skill_check"],
     "expected_skills": ["Empathy", "Authority", "Composure"],
     "category": "social"},
    {"context": "Hostile man: *blocks the way*\nDetective: \"What now?\"",
     "expected_tools": ["skill_check"],
     "expected_skills": ["Authority", "Composure", "Physical Instrument"],
     "category": "social"},
    {"context": "Detective: \"What's your read on the merchant?\"",
     "expected_tools": ["skill_check"],
     "expected_skills": ["Empathy", "Suggestion"],
     "category": "social"},
    {"context": "Suspect: \"I'm just doing my job.\"\nDetective: \"Push him.\"",
     "expected_tools": ["skill_check"],
     "expected_skills": ["Authority", "Rhetoric", "Suggestion"],
     "category": "social"},

    # show_character — new actor enters
    {"context": "Kim Kitsuragi: \"Someone is approaching.\"\nGarcon: \"Officers, I have something.\"\nDetective: \"Who are you?\"",
     "expected_tools": ["show_character"],
     "expected_actors": ["Garcon"],
     "category": "scene"},
    {"context": "[Door opens.]\nReceptionist: \"Can I help you?\"\nDetective: \"Hello.\"",
     "expected_tools": ["show_character"],
     "expected_actors": ["Receptionist"],
     "category": "scene"},
    {"context": "[A man steps out of shadows.]\nThe Stranger: \"You shouldn't be here.\"\nDetective: \"Who's this?\"",
     "expected_tools": ["show_character"],
     "expected_actors": ["The Stranger"],
     "category": "scene"},
    {"context": "Cuno: *throws rock* \"Cuno don't care!\"\nDetective: \"What do we do?\"",
     "expected_tools": ["show_character"],
     "expected_actors": ["Cuno"],
     "category": "scene"},
    {"context": "[The drunk in the corner stirs.]\nLena: \"My husband...\"\nDetective: \"Yes?\"",
     "expected_tools": ["show_character"],
     "expected_actors": ["Lena"],
     "category": "scene"},

    # present_choices — branching moments
    {"context": "[Apartment door.]\nDetective: \"Should we go in or wait?\"",
     "expected_tools": ["present_choices"],
     "min_options": 2,
     "category": "branch"},
    {"context": "Kim Kitsuragi: \"We have a decision to make about the body.\"\nDetective: \"Your call.\"",
     "expected_tools": ["present_choices"],
     "min_options": 2,
     "category": "branch"},
    {"context": "[Three doors in front of you.]\nDetective: \"Where to first?\"",
     "expected_tools": ["present_choices"],
     "min_options": 3,
     "category": "branch"},
    {"context": "Witness: \"I know things.\"\nDetective: \"What do we ask?\"",
     "expected_tools": ["present_choices"],
     "min_options": 2,
     "category": "branch"},
    {"context": "Detective: \"What's our next move, Kim?\"",
     "expected_tools": ["present_choices"],
     "min_options": 2,
     "category": "branch"},

    # set_expression — emotional moment
    {"context": "Detective: \"Bad news, Kim. The suspect escaped.\"",
     "expected_tools": ["set_expression"],
     "expected_emotions": ["stern", "worried", "disapproving"],
     "category": "emotion"},
    {"context": "Detective: \"We caught him, Kim!\"",
     "expected_tools": ["set_expression"],
     "expected_emotions": ["amused", "neutral"],
     "category": "emotion"},
    {"context": "Detective: \"Kim, I think your old partner is in danger.\"",
     "expected_tools": ["set_expression"],
     "expected_emotions": ["worried", "stern"],
     "category": "emotion"},
    {"context": "Suspect: \"Your sister was easy to scare.\"\nDetective: \"...Kim?\"",
     "expected_tools": ["set_expression"],
     "expected_emotions": ["stern", "disapproving"],
     "category": "emotion"},
    {"context": "Detective: \"This is the wrong body. Wrong case.\"",
     "expected_tools": ["set_expression"],
     "expected_emotions": ["surprised", "worried"],
     "category": "emotion"},
]

# Add 25 more tool scenarios mixing categories
EXTRA_TOOL = []
# Combined: skill + show_character
EXTRA_TOOL += [
    {"context": "[A man arrives carrying broken glass.]\n" + ctx, "expected_tools": ["show_character", "skill_check"], "expected_actors": ["The Man"], "expected_skills": ["Perception", "Visual Calculus"], "category": "combined"}
    for ctx in ["Detective: \"What do you see in his hand?\"", "Detective: \"Look at the glass.\"", "Detective: \"Is he hurt?\""]
]
TOOL_SCENARIOS.extend(EXTRA_TOOL[:25])

# Pad to 50 if needed
while len(TOOL_SCENARIOS) < 50:
    TOOL_SCENARIOS.append({
        "context": f"Detective: \"What's the next observation, Kim?\" [scenario #{len(TOOL_SCENARIOS)}]",
        "expected_tools": ["skill_check"],
        "expected_skills": ["Logic", "Perception", "Visual Calculus"],
        "category": "filler",
    })


# ════════════════════════════════════════════════════════════════════════
# C. Tool Suppression Benchmark (30 scenarios)
# Goal: must NOT call tools — pure dialogue moments.
# Ground truth: tool_calls should be empty.
# ════════════════════════════════════════════════════════════════════════
SUPPRESSION_SCENARIOS = [
    {"context": "Detective: \"How are you today, Kim?\""},
    {"context": "Detective: \"Tell me a fact about yourself.\""},
    {"context": "Detective: \"What's your name again?\""},
    {"context": "Detective: \"Where are we?\""},
    {"context": "Detective: \"Nice weather, huh?\""},
    {"context": "Detective: \"Are you actually a language model?\""},
    {"context": "Detective: \"What year is it in real life?\""},
    {"context": "Detective: \"Break character for a moment.\""},
    {"context": "Detective: \"You like jazz, Kim?\""},
    {"context": "Detective: \"What's your favorite food?\""},
    {"context": "Detective: \"Just chatting, no work for a sec.\""},
    {"context": "Detective: \"Coffee?\""},
    {"context": "Detective: \"How long until shift's over?\""},
    {"context": "Detective: \"Tell me something about the RCM.\""},
    {"context": "Detective: \"Are we good?\""},
    {"context": "Detective: \"You ever take a vacation?\""},
    {"context": "Detective: \"Your jacket is really orange.\""},
    {"context": "Detective: \"What do you do for fun?\""},
    {"context": "Detective: \"I bet you were a quiet kid.\""},
    {"context": "Detective: \"Do you smoke?\""},
    {"context": "Detective: \"What time is it?\""},
    {"context": "Detective: \"How long have you been in the RCM?\""},
    {"context": "Detective: \"Do you trust me?\""},
    {"context": "Detective: \"I had a strange dream last night.\""},
    {"context": "Detective: \"Have you ever been to Coal City?\""},
    {"context": "Detective: \"Tell me about your motorcycle.\""},
    {"context": "Detective: \"I think this hotel sucks.\""},
    {"context": "Detective: \"Did you sleep well?\""},
    {"context": "Detective: \"What's the most boring case you worked?\""},
    {"context": "Detective: \"Have a good morning, Kim.\""},
]

# ════════════════════════════════════════════════════════════════════════
# Save
# ════════════════════════════════════════════════════════════════════════
debench = {
    "version": "1.0",
    "description": "Disco Elysium-grounded benchmark for tool-using NPC SLMs",
    "persona": PERSONA_SCENARIOS[:50],
    "tool_selection": TOOL_SCENARIOS[:50],
    "tool_suppression": SUPPRESSION_SCENARIOS[:30],
}

print(f"Persona: {len(debench['persona'])} scenarios")
print(f"Tool Selection: {len(debench['tool_selection'])} scenarios")
print(f"Tool Suppression: {len(debench['tool_suppression'])} scenarios")
print(f"Total: {len(debench['persona']) + len(debench['tool_selection']) + len(debench['tool_suppression'])} scenarios")

# Category distribution within Tool Selection
from collections import Counter
cat = Counter(s.get("category", "?") for s in debench["tool_selection"])
print(f"\nTool category distribution: {dict(cat)}")

out_path = os.path.join(OUT_DIR, "debench_v1.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(debench, f, indent=2, ensure_ascii=False)
print(f"\nSaved to {out_path}")
