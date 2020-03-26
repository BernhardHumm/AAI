
# Complex Event Processing

*Complex event processing (CEP)* allows processing streams of event data and deriving conclusions from them.

Fig. 8.1. shows CEP in the AI application landscape.

![Fig. 8.1: CEP in the AI landscape](images/AI_landscape-CEP.png)

CEP can be assigned to the ability of "reasoning". 


Examples of CEP applications are:

1. Detecting fraud based on payment transaction data
2. Making selling / buying decision based on stock market feeds
1. Switching active traffic signs based on traffic data
1. Making purchase recommendations based on click stream analysis in webshops

All those appications have in common that complex decisions (e.g., fraud alert, selling / buying etc.) are made in real-time based on events (e.g., payment transactions, stock market feeds etc.). 


## Foundations

What is an *event* in this context?

An event is something notable that happens.

What is notable or not entirely depends on the application use case. 
Examples are:

1. A financial transaction
1. An airplane landing
1. A sensor outputs a reading
1. A change of state in a database, a finite state machine
1. A key stroke
1. A historic event, e.g., the French Revolution

An event takes place in the real world.
An *event object* is a data record that represents an event.

Examples are:

1. A Purchase order record
1. A stock tick message
1. A record with an RFID sensor reading


An *event type (a.k.a. event class)* specifies the common structure of related event objects, i.e., their attributes and data types, e.g., `PurchasingEvent` with attributes `timestamp`, `buyer`, `product` and `price`.

In CEP literature, often the term "event" is also used for event objects and event types. From the context it is usually clear whether the real-world event is meant the concrete data record or its type.

A *CEP engine* allows specifying *CEP rules* in a *CEP rule language* and executing them. CEP rules specify patterns that can match events in an *event stream*. *Message brokers* are platforms to manage streams of messages, in particular event objects. 



Fig. 8.2. illustrates the difference between data access in database management systems (DBMS) and CEP engines.

{width=60%}
![Fig. 8.1: Data access in (a) DBMS versus (b) CEP engine](images/AI_landscape-CEP.png)


A DBMS stores persistent data. A query, e.g., formulated in SQL, executes instantaneously and returns a query result based on the current state of the persistent data.
In contrast, the data source for CEP is a flowing stream of events. CEP rules are persistent, with the CEP engine constantly trying to match CEP rules with the events. Whenever a CEP rule matches, a higher (complex) event is being generated which may trigger certain actions. 




## Application Example: Fault Detection in the Smart Factory

I will explain CEP with the application example of one of my research projects (Beez et al., 2018): fault detection in the smart factory. 







(Kaupp et al., 2017)









## Services Map and Product Map

Fig. 8.x shows the CEP services map. 

{width=75%}
![Fig. 8.x: CEP Services Map](images/CEP_SM.png)


Message brokers implement a message-oriented middleware, providing message queuing technology. The allow managing streams of event messages. They also provide consoles for administering message queues. 
CEP engines allow specifying CEP rules and executing them. They also allow monitoring CEP.  



Fig. 8.x shows the CEP product map. 


{width=75%}
![Fig. 8.x: CEP Product Map](images/CEP_PM.png)

All major IT vendors provide message brokers and CEP engines. Also, there are several production-ready open source solutions. 
Examples of message brokers are Apache Kafka, IBM MQ, TIBCO, WebSphere Business Events, and WSO2 Stream Processor.
Examples of CEP engines are Apache Flink,  MS Azure Stream Analytics, Drools Fusion, Oracle Stream Analytics, and SAG Apama.




## Quick Check

X> Answer the following questions.

1. What is complex event processing (CEP)?
1. What is an event, event object and event type?
2. What is a message broker?
2. What is a CEP engine? 
1. Explain the difference between data access in a DBMS versus a CEP engine?
1. Name prominent CEP products

