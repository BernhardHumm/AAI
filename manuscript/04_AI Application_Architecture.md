

# AI Application Architecture


What characterizes an AI application?  Is it the use of a particular technology like LLMs, ML libraries, a rule engine or an agent framework? 
Or is it maybe a particular architectural choice like a blackboard architecture?

In my opinion, it is not the implementation characteristics that are relevant for characterizing an AI application.
It is mainly the application use case that is relevant:
AI applications exhibit behavior of human intelligence - whatever is under the hood. 

AI applications are IT applications, no matter which technology is used. Traditional software engineering principles and techniques apply, including separation of concerns, information hiding, layering, component-orientation, etc. Common issues like user experience, performance, security, maintainability, and cost-effectiveness are important. Additionally, a sound development methodology is important. When following an agile development approach, this includes e.g., early releases and prototypes, regular user feedback and quality assurance, etc.

To rephrase: *AI applications are IT applications and therefore classic software engineering principles apply*. 

%% This book focuses on engineering AI applications of high quality in an effective, cost-efficient way. 



## AI Reference Architecture

A *reference architecture* is a blueprint for concrete application architectures. In a reference architecture, the benefits of many concrete application architectures are distilled. The architect in a development project may use a reference architecture as a starting point for developing the concrete application architecture. 

Fig. 4.1 shows a reference architecture for AI applications. 

![Fig. 4.1: AI reference architecture](images/AI_Reference_Architecture.png)


The reference architecture is structured as a *layered architecture* similar to the classic three-layer-architecture of business information systems. 

The *interaction layer* implements the (graphical) user interface which, in the case of a web apps or mobile apps, may be implemented with state-of-the-art UI frameworks based on HTML/CSS/JavaScript. The user interaction can be multi-modal based on voice, text, images or videos. In an embedded AI system like a mobile robot, physical actions may also be performed. 

A commonly used metaphor for the application logic in an AI system is that of an *AI agent*. Python is the de-facto standard programming language for AI applications, but all other state-of-the-art programming languages like Java, C# and C++ are also in use. Often, the agent logic (the controller of the AI application) is hard-coded in the programming languages. If more flexibility is required, agent frameworks can be used, e.g., LangChain and LlamaIndex for LLM-based AI agents.

AI applications typically base their intelligent behavior on AI resources like ML models or knowledge graphs. ML models may be custom models trained for the purpose of the AI application. In enterprises, often ML *lifecycle management* systems (e.g., mlflow) are used that manage ML training, model versioning, validation and monitoring as well as model deployment.

Often, off-the-shelf ML models are used, e.g., LLMs, usually fetched from *AI repositories* like huggingface. They provide all kinds of pre-trained ML models. Also, off-the-shelf knowledge graphs like WikiData or Word Nets can be obtained from public sites.

In contrast to loading AI resources from repositories, *AI web services* may be accessed from cloud providers like Google, Amazon, Microsoft or IBM. They include AI services for NLP (e.g., speech-to-text, text-to-speech, chatbots), CV (e.g., image or video generation), ML (e.g., training and deployment of large ML models) and orchestration (e.g., workflows or agents).

AI resources may also knowledge-graphs or wordnets as well as hybrid resources like vectorstores. 



The *application logic layer* implements the intelligence of the AI application, e.g., in form of intelligent agents. 
The AI application logic is usually implemented in a general-purpose programming language. Within the last years, Python is evolving as the primary AI programming language (as Lisp and Prolog were in the 1980s). But object-oriented languages like Java, C# and C++ are also in common use. 
Often, powerful libraries and frameworks for AI tasks like machine learning, language processing, image processing, etc. are utilized. Examples are Keras, TensorFlow, Scikit-learn and Spark. 

Instead of including AI libraries, 3rd party AI web services can be used. Major vendors like Google, Amazon, Microsoft, and IBM offer AI suites with web services for machine learning, language processing, image processing etc.

The underlying data of the AI application is stored in a *knowledge base* which is accessed by the application logic layer via an API. As outlined in the chapter on knowledge representation, technologies with reasoning engines (like, e.g., Apache Jena) may be used. However, also classic storage technologies like RDBMS or NoSQL databases are often used.  

Finally, data may be loaded into the knowledge base from various sources, e.g., knowledge graphs, databases, web pages, documents etc. Those data may be *integrated and semantically enriched* (see the chapter on knowledge representation).

In a concrete AI application, each of those layers may be developed differently. Also, depending on the application use case, individual layers may be missing entirely. For example, in applications where knowledge items are created by the users, the data integration layer is not needed. In an application where the reasoning capabilities of the knowledge base are sufficient, an explicit application logic layer may be omitted. In an embedded AI application, e.g, a robot, a graphical user interface is not needed.  





## Application Example: Virtual Museum Guide

Let us consider a concrete application scenario: a *virtual museum guide*. The task of the virtual museum guide is to guide users through a virtual museum - much like a human museum guide who guides visitors though the physical museum. 

The degree of intelligence of this virtual museum guide may vary a lot. 
In the simplest form, the guide may offer the users displays of paintings along with descriptive text. 
The paintings may simply be presented in a fixed order which has been curated by a human. 


In the most complex form, the virtual museum guide tells stories about the paintings, answers natural-language questions of the users (possibly via speech input and output), and adapts the selected paintings and stories to the users' background. For example, children will be told different stories than adults.

While one would probably not consider the simple virtual museum guide as intelligent, the complex one definitely exhibits behavior of human intelligence: understanding, speaking, telling stories, answering questions, etc.

Fig. 4.2 shows a potential architecture of a virtual museum guide application. 

{width=75%}
![Fig. 4.2: Example architecture: Virtual museum guide application](images/Virtual_Museum_Guide_Application_Architecture.png)

In this architecture, the virtual museum guide GUI is implemented with HTML5 / CSS and JavaScript including  state-of-the-art libraries. The application logic of the virtual museum guide is implemented in Java including libraries like Eclipse rdf4j. Sub-components are:

- Arts model: for representing artworks and their content
- User model: for representing the current user and his / her background
- Story generation: for generating stories about artworks suitable for the current user
- Natural language processing (NLP): for generating voice output and analyzing voice input

The knowledge base is implemented using Eclipse RDF4J (API and knowledge base including reasoner and SPARQL query engine). The Art knowledge graph is loaded into RDF4J at system start. In an offline step, it is extracted beforehand via Python scripts from Wikidata.



## Data Integration / Semantic Enrichment

The data integration aspect is, in my opinion, not treated enough in AI literature. 
The knowledge in AI applications often stems from various data sources (see the chapter on knowledge representation). This is similar in the field of Business Intelligence (BI) where the process of integrating data from various sources into a data warehouse (DWH) is often called *ETL (Extraction, Transformation, Loading)*. 

ETL can be seen as an architectural pattern where business information systems (data sources) are separated from business intelligence systems. ETL is a pipeline for extracting, transforming, and loading data in a format highly optimized for its use (analytic application).

The ETL architectural pattern also is suitable for the data integration of AI applications. Since in AI applications, data is often semantically enriched, I use the term *Semantic ETL*.

Semantic ETL consists of the following steps.

1. *Extraction* of data from source systems: These could be files, websites, databases, SPARQL endpoints, etc., e.g., the DBpedia SPARQL endpoint. 
2. *Filtering* irrelevant data and data of insufficient quality; e.g., selecting  only paintings, sculptures and the respective artists from Wikidata; selecting  English descriptions only  and filtering attributes with wrong datatypes.
3. *Technical format transformation*: transforming from the source formats to the target format, e.g., from JSON to RDF
4. *Data schema transformation*: transforming from the data schemas of the source format to a target data schema, e.g., renaming `wd:Q3305213` to `:artwork`
5. *Semantic enrichment*: heuristically integrating semantic information from various data sources, e.g., Michelangelo's birth and death date from GND, his influences from YAGO, and his paintings from Wikidata
6. *Performance tuning*: optimizing the data storage according to the application use cases, e.g., normalizing data and indexing for high-performance access
7. *Loading*: storing data in the target knowledge base, e.g., rdf4j.




## Presentation

The (graphical) user interface of an AI application is not AI-specific. As in all IT applications, it is of essential importance for the user's experience of the application. See the comprehensive literature for developing user-centric applications.  



## Programming Languages

Within the last years, Python is being established as the major AI programming language. This development has been supported by major players publishing their AI libraries and frameworks in Python, e.g., Google TensorFlow.
Also, there are still numerous AI libraries available in traditional object-oriented programming languages like Java, C#, and C++. The traditional AI programming languages of the 1980, Lisp and Prolog, only play a niche role in today's AI application development. But they have a major influence on the design of modern dynamic programming languages like Python, R, Julia and others.

For making a sound programming language decision in an AI application development project, one should consider various aspects:

- Which technology stack offers the best resources (runtime platform, libraries, developers' tools, etc.)?
- Are there enough developers familiar with the technology?
- Where is the best support (User groups etc.)

To conclude: for developing AI applications, all software engineering principles for the development of large-scale, complex IT systems apply. 






## Application Logic /  Agents

In many AI publications, e.g., (Russell and Norvig, 2021), *agents* are described as a metaphor for the central component of an AI application which exhibits intelligent behavior.

Fig. 4.3 by Russell and Norvig (1995) illustrates the concept of an agent.

![Fig. 4.3: The concept of an agent (Russell and Norvig, 1995, Fig. 2.1)](images/Agent.png)

An agent interacts with an environment and via sensors it perceives the environment. 
The agent logic then reasons over its perceptions and its internal expert knowledge and plans respective actions. Via actuators  it executes those actions. The executed actions may, in turn, have an effect on the environment which is perceived, again, by the agent.

Fig. 4.4 shows examples of agents, from simple to complex.

![Fig. 4.4: Examples of agents](images/Examples_of_Agents.png)

![Fig. 4.4: Agentic cycle: perceive, reason, act](images/Agent_cycle.png)

FIG PERCEIVE/REASON/ACT


Is it appropriate to call the virtual museum guide an agent?
In its simple form (pre-defined guides) one might intuitively say "no".
In its complex form (story telling) the answer surely is "yes". 

In my opinion, however, this question is not too relevant.  Much more relevant is the question whether the agent metaphor is beneficial for designing the virtual museum guide application. And this question may well be answered with "yes". 
Thinking of the virtual museum guide as an agent may lead to an architecture where perceptions are separated from actions and where the agent establishes a model of all past perceptions that is used for planning the next actions. And the separation of those concerns may well be a good architectural decision. 


DELETE

%%![A Simple Agent (Source: Stuart Russel, Berkeley, Lecture on Artificial Intelligence)](images/Simple_Agent.png)
%%![A Complex Agent (Source: Stuart Russel, Berkeley, Lecture on Artificial Intelligence)](images/Complex_Agent.png)





### Agent Frameworks

*Agent frameworks* provide a base architecture and offer services for developing the agent logic of an AI application. 
A number of agent frameworks implement a plug-in architecture where framework components and custom components can be integrated. Some frameworks specify domain-specific languages (DSL) for the agent logic. Usually, APIs for integrating code in different programming languages are provided. 

See, e.g., the architecture of [Cougaar](http://www.cougaar.world) in Fig. 4.5.


#### Knowledge-Based Agent Frameworks


![Fig. 4.5: An agent framework example: Cougaar (More et al., 2004)](images/Agent_Framework.png)


In Cougaar, coordinator components like a Cost/Benefit Plugin provide the agent logic. The blackboard component is a shared repository for storing information on current problems, suggestions for solving the problems as well as (partial) solutions. 
Sensors and actuator components may be plugged in. Sensors regularly update  information on the blackboard.

Other agent frameworks are 
[JaCaMo](http://jacamo.sourceforge.net),
[JADE](http://jade.tilab.com),
[JIAC](http://www.jiac.de/agent-frameworks),
[AgentFactory](https://sourceforge.net/projects/agentfactory), and the
[Jadex BDI Agent System](http://sourceforge.net/projects/jadex/).
For details see the appendix.



### When to use an Agent Framework?

In my opinion, the metaphor of an agent is useful when designing an AI application. 
The separation of sensors from actuators as well as the separation of a model of the application domain and the environment from the agent logic is good architectural practice.
However, the use of an agent framework is not always recommended. This is because every framework involves a learning curve and adds new  technological dependencies to the project. 

If the agent logic is sufficiently complex and the services provided by the framework are suitable for the application use case then the costs for introducing an agent framework may well be justified.
However, if this is not the case, a traditional component-based software architecture is adequate. The architectural recommendations following the agent metaphor can still be implemented to some degree. 



#### LLM-Based Agent Frameworks


#### Example: LangChain Agents


![Fig. 4.4: LangChain agent framework example](images/LangChain_agent.png)

Tutorial https://python.langchain.com/docs/tutorials/agents 


    # Import relevant functionality
    from langchain.chat_models import init_chat_model
    from langchain_tavily import TavilySearch
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.prebuilt import create_react_agent

    # Create the agent
    memory = MemorySaver()
    model = init_chat_model("anthropic:claude-3-5-sonnet-latest")
    search = TavilySearch(max_results=2)
    tools = [search]
    agent_executor = create_react_agent(model, tools, checkpointer=memory)

Using the agent:


    # Use the agent
    config = {"configurable": {"thread_id": "abc123"}}

    input_message = {
        "role": "user",
        "content": "What's the weather in SF?",
    }
    for step in agent_executor.stream(
        {"messages": [input_message]}, config, stream_mode="values"
    ):
        step["messages"][-1].pretty_print()



Result:


    ================================[1m Human Message [0m=================================

    What's the weather in SF?
    ==================================[1m Ai Message [0m==================================

    [{'text': 'Let me search for current weather information in San Francisco.', 'type': 'text'}, {'id': 'toolu_011kSdheoJp8THURoLmeLtZo', 'input': {'query': 'current weather San Francisco CA'}, 'name': 'tavily_search', 'type': 'tool_use'}]
    Tool Calls:
    tavily_search (toolu_011kSdheoJp8THURoLmeLtZo)
    Call ID: toolu_011kSdheoJp8THURoLmeLtZo
    Args:
        query: current weather San Francisco CA
    =================================[1m Tool Message [0m=================================
    Name: tavily_search

    {"query": "current weather San Francisco CA", "follow_up_questions": null, "answer": null, "images": [], "results": [{"title": "Weather in San Francisco, CA", "url": "https://www.weatherapi.com/", "content": "{'location': {'name': 'San Francisco', 'region': 'California', 'country': 'United States of America', 'lat': 37.775, 'lon': -122.4183, 'tz_id': 'America/Los_Angeles', 'localtime_epoch': 1750168606, 'localtime': '2025-06-17 06:56'}, 'current': {'last_updated_epoch': 1750167900, 'last_updated': '2025-06-17 06:45', 'temp_c': 11.7, 'temp_f': 53.1, 'is_day': 1, 'condition': {'text': 'Fog', 'icon': '//cdn.weatherapi.com/weather/64x64/day/248.png', 'code': 1135}, 'wind_mph': 4.0, 'wind_kph': 6.5, 'wind_degree': 215, 'wind_dir': 'SW', 'pressure_mb': 1017.0, 'pressure_in': 30.02, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 86, 'cloud': 0, 'feelslike_c': 11.3, 'feelslike_f': 52.4, 'windchill_c': 8.7, 'windchill_f': 47.7, 'heatindex_c': 9.8, 'heatindex_f': 49.7, 'dewpoint_c': 9.6, 'dewpoint_f': 49.2, 'vis_km': 16.0, 'vis_miles': 9.0, 'uv': 0.0, 'gust_mph': 6.3, 'gust_kph': 10.2}}", "score": 0.944705, "raw_content": null}, {"title": "Weather in San Francisco in June 2025", "url": "https://world-weather.info/forecast/usa/san_francisco/june-2025/", "content": "Detailed ⚡ San Francisco Weather Forecast for June 2025 - day/night 🌡️ temperatures, precipitations - World-Weather.info. Add the current city. Search. Weather; Archive; Weather Widget °F. World; United States; California; Weather in San Francisco; ... 17 +64° +54° 18 +61° +54° 19", "score": 0.86441374, "raw_content": null}], "response_time": 2.34}
    ==================================[1m Ai Message [0m==================================

    Based on the search results, here's the current weather in San Francisco:
    - Temperature: 53.1°F (11.7°C)
    - Condition: Foggy
    - Wind: 4.0 mph from the Southwest
    - Humidity: 86%
    - Visibility: 9 miles

    This is quite typical weather for San Francisco, with the characteristic fog that the city is known for. Would you like to know anything else about the weather or San Francisco in general?



LLM performs reasoning, tools perform perception and action. 
here: out of the box.
via prompt engineering you can customize reasoning.
you can also implement own tools, e.g., for accessing APIs, e.g., local databases, but also web services.

PROTOCOL.


Multi-agent



#### Engineering Agent Systems


Define objectives
choose agent roles
select the right framework
develop communication protocols
implement decision-making models
test and iterate


#### LLMs and Reasoning

Are LLMs really reasoning?

-----------------------------------------


You're absolutely right to question this—it's one of the most debated topics in AI today. The short answer is: **LLMs simulate reasoning**, but whether they *truly* reason depends on how you define it.

---

### 🧠 What LLMs Actually Do

LLMs like GPT or Claude generate text by predicting the **most probable next token** given a context. This is statistical pattern matching, not symbolic logic. They don’t have:
- A formal model of logic or truth.
- Internal representations of beliefs or goals.
- Deductive or inductive reasoning in the classical sense.

So yes—at their core, they’re **probabilistic sequence generators** trained on vast corpora.

---

### 🤔 But Then Why Do They *Look* So Smart?

Because:
- They’ve seen millions of examples of reasoning, argumentation, and problem-solving.
- They can **mimic reasoning patterns**—like solving math problems, writing proofs, or making decisions—by reproducing structures they've learned.
- They can chain thoughts, revise answers, and even self-correct, which *feels* like reasoning.

This leads to **emergent behavior**: they appear to reason, even though they don’t have a reasoning engine.

---

### 🧩 So Are They Reasoning?

Let’s break it down:

| Perspective         | Verdict on LLM Reasoning           |
|---------------------|------------------------------------|
| **Philosophical**   | No—they lack intentionality or understanding. |
| **Cognitive science** | Maybe—they simulate reasoning behavior. |
| **Engineering**     | Yes—they solve problems effectively. |
| **Logic/AI**        | Not really—they don’t use formal inference. |

---

### 🧠 What *Is* Real Reasoning Then?

Traditional reasoning involves:
- **Deduction**: From rules to conclusions.
- **Induction**: From examples to generalizations.
- **Abduction**: From effects to possible causes.

LLMs don’t do this explicitly. But they *approximate* it by pattern matching across billions of examples.

---

### 🔮 The Future: Hybrid Reasoning

The most promising direction is **hybrid systems**:
- LLMs for flexible, fuzzy reasoning.
- Symbolic engines (like Prolog, SAT solvers, or rule-based systems) for precision.
- Agents that combine both—reasoning when needed, generating when helpful.

------------------------------------------


#### Model Context Protocol (MCP)


#### Example: GitHub Copilot


## hybrid AI


## Quick Check

X> Answer the following questions.

1. What characterizes an AI application? 
1. What are the main components of the AI reference architecture?
1. Could you speak of an AI application  if none of these technologies are used: reasoning engine, machine learning framework, agent framework?
1. What is an agent? Give examples.
1. Which services do agent frameworks offer?
1. In which situations is the use of an agent framework recommended? In which not?
2. Which programming language should be used for developing an AI application?

	
