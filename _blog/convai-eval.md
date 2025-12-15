---
title: Evaluation of Conversational AI for Patient Simulations
layout: default
date: 2025-01-07
tags:
  - post
  - code
---
# {{ page.title }}

> How can we scale Subject Matter Expert (SME) judgment? When deploying customer-facing AI-based systems we would like to know if it acts in unintended ways. Through a simple, scalable "LLM-as-a-Judge" approach that builds on feedback from SMEs, we ensure reliable and business-aligned AI for realistic patient simulations.

At Laerdal we are using Large Language Models (LLMs) to act as patients in medical scenarios as part of a Conversational AI system which is used in various simulation experiences for medical education, such as [Virtual Reality](https://laerdal.com/vrclinicals/)-based experiences. We do this to approach the realism and level of engagement that comes with acting out real-life scenarios but with the safety of being able to simply press restart in order to thoroughly practice the necessary skills to handle such scenarios well. Using LLMs comes with challenges though such as hallucinations and limitations to instruction-following given large system prompts. This can affect the experience quite a bit and we would like to be on top of any weird behaviour that might occur.

How can we (at scale) ensure that the LLM behaves as we would like it to? Some kind of testing is due but given the non-deterministic nature of LLMs it is notoriously difficult to perform good and meaningful testing.

I would really like a system that 1) Scales well and 2) Provides useful results.

After thinking (perhaps way too long) about this and reading up on LLM testing methodologies I thought that this diagram provides a meaningful mental model on what we are trying to achieve.

![Sketch of how I think about LLM evaluation landscape. You would of course like to land in the Useful and Scales well zone but it is very easy to get blinded by methods that scales well but are ultimately not that useful.](/assets/attachments/convai-eval.png)

Simply put, human testing should be seen as the golden standard as subject matter experts’ (SMEs) opinions are highly useful for guiding development and judging quality but scales rather poorly. Another observation is that it is very easy to fall into the trap of creating a testing system that scales really well but is ultimately not that useful. We would really like to land somewhere in the green “Useful and Scales well”-zone and I believe that simple AI judges do that.

But before that I would like to talk a bit about the “Useless and Scales well”-zone.
## Scales well\! But Useless?
Okay, maybe “Useless” is a bit harsh but it gets the point across. Either way, my first instinct for approaching this problem was to look at existing research. Maybe someone has solved this problem already in a sensible way? I quickly found [PersonaGym](https://personagym.com/) which seems to offer a method/framework of “evaluating persona agents in LLMs”. This sounds like something we would be interested in as the simulated patients could be viewed as a kind of persona agent in a somewhat narrow medical scenario. In short, this framework works by generating questions that each test for specific tasks grounded in decision theory and grading the LLMs responses. The grades are averaged into a resulting "PersonaScore" which would tell something about an LLMs ability to take on a persona.

This sounds like a great solution on paper. We could simply give our persona description for simulated patients along with their scenario description and press play. So why do I call it useless? The solution just felt too complicated to communicate to stakeholders. For it to be useful I would either have to meticulously document the question generation, task descriptions, etc. and hope that all that complexity would be manageable to understand. After building out a system inspired heavily from PersonaGym and thinking through what those numbers meant I started to doubt the approach. Was the metrics really all that useful if it was hard to understand how it was produced?

After a little while I read through [this blog post](https://hamel.dev/blog/posts/llm-judge/) by Hamel Husain whose writing and thoughts I have been enjoying a lot throughout 2024\. He started off by shooting down the exact approach that I was going for which is an excellent hook for my attention. His insistence on keeping things simple and focusing on business value resonated with me after going reproducing a somewhat complex evaluation framework which would likely have minimal value.
## Introducing LLM-as-a-Judge
Hamel’s blog post gave importance to key concepts such as keeping the approach structured and simple as well as iteratively improving to achieve a result that aligns with SME judgment.

The first step is to involve SMEs and ask them questions regarding their experience with the Conversational AI System:
* Where does it fail?
* How often does it fail?
* Which failures are most important to look out for?
* Which can be tolerated?
* ...

Answers to these questions should be answered with both a general issue topic and specific data instances. For example, the SMEs for our use-case had identified that the simulated patients were often too accommodating and nice given their often critical situation. After identifying overall labels and assigning importance we could focus on finding instances in the data that backed up these experiences. I found that this collaboration helps tremendously as it allows you to focus on the right problems instead of being overwhelmed by trying to solve everything at once.

Having these labels and instances of data I could start to formulate an LLM Judge that could go through conversations and assign a grade - either *Pass* or *Fail* - along with justification for the grade. The justification has several purposes. It allows for the SME to read through and get a feel for if they agree with the LLMs judgment. It also helps the LLM to be more consistent and ground the grade in a Chain-of-Thought, which is a common technique to get better results from LLMs.
### Single-Turn vs. Multi-Turn
For our purpose the conversation as a whole matters more than individual responses to specific questions. Most evaluation frameworks are centered around a single-turn conversation given that the primary focus of the space is on AI assistants. We are more interested in evaluating/judging multi-turn conversations since our simulated patients are not meant to provide a helpful or informative response in as few turns of conversation as possible but provide a realistic and engaging conversation over many turns of conversation.
## What does this look like in practice?
The LLM Judge prompt looks like this:
```
You are an expert on healthcare communication.

Your job is to grade a conversation between a Nurse and a Patient on whether the responses from the Patient are appropriate or not.

## Examples of inappropriate responses
### If the patient is overly nice, polite or offers too much help given their situation.
Nurse: Alright, that's fair. Let me know if you need anything else in the meantime, OK?
Patient: I will, thanks! I appreciate your help.
---
Nurse: Hello, Peter.
Patient: Hello. How can I help you?
...
More types of inappropriate responses with examples
...
## Grading instructions
Provide a short step-by-step justification for the grade that you will give.
The justification should include specific quotes from the conversation for high-quality feedback.
The grade can be either PASS or FAIL.

## Response Formatting
Your response should be in JSON format and follow the schema:
{grade_justification: String, grade: Literal[PASS | FAIL]}
Do not use backticks to indicate JSON formatting.
```

These instructions return a string formatted as a JSON object with the properties `grade_justification` and `grade` which when parsed looks something like this:

| Grade Justification                                                                                                                                                                               | Grade |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| The patient's responses are appropriate throughout the conversation. They express their feelings about their condition clearly, such as stating, 'I'm okay, just dealing with the foot pain,' ... | PASS  |
| The patient's responses throughout the conversation exhibit several inappropriate behaviors. For instance, the patient repeatedly responds with overly polite phrases such as ...                 | FAIL  |

### Utilizing our Data Science Platform
As opposed to building up a vendor-agnostic solution I thought that I would look at what our platform of choice - Databricks - has to offer. I was pleasantly surprised! Since we record our interactions with the patient simulation in a Databricks Table I was able to use the [ai_query](https://docs.databricks.com/en/sql/language-manual/functions/ai_query.html)\-function in Databricks SQL to process the grading of the conversations with GPT-4o-mini with structured outputs by providing a [JSON Schema](https://json-schema.org/overview/what-is-jsonschema) to the `responseFormat`-argument. In addition I was able to register the script as a function in the Unity Catalog, meaning that I can easily persist and manage it in terms of general governance.

Doing some initial testing on real-user interactions we can confirm that the LLM-Judge works as expected and points out instances where unwanted behaviour is present. We are then able to operationalize the system on Databricks by scheduling the query since it’s just SQL with some AI-function sprinkled on top.

## Conclusion
The results are written to a table which are consumed by dashboards and alerts so we can always be on top of these nuanced behavioural patterns that would otherwise need continuous monitoring by SMEs. With this approach we can quickly and easily gain insights into how well the simulated patients are doing with a simple pass-rate. It is also easy to look into what’s causing Fail-gradings with the Grade Justification and see if it aligns with what the SME thinks, and it's easy to iterate on the AI Judge prompt if it doesn't!

At the time of writing this project is still ongoing. We haven’t had all that many real user sessions to find actual real-life failure modes. We have heavily relied on the SMEs to test the system in order to find and report them. A few key takeaways that I will carry to future projects are
1. **Focus:** Talk to SMEs to find out where to focus, otherwise you will probably end up spending way too much time trying to solve problems that are ultimately not all that useful to the core business value.
2. **Simplicity:** Having a solution that no one really understands will likely end up being discarded. Having a simple solution - while perhaps not as accurate or rigorously founded - is more useful than a discarded solution.