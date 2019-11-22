# Initial client request

``` mermaid
graph LR;
A[Client]-->|Initial request with channel on which client will be listening|B[Redis]
B-->|Node gets message from queue|C[Node]
C-->|set itself as control node|C
  ```
# Establishing committee
``` mermaid
sequenceDiagram;
  participant ControlNode
  participant Redis
  participant FreeNodes
  ControlNode ->> Redis:Send create committee event with control node listen channel
  Redis ->> FreeNodes: Notify
  FreeNodes ->> Redis: Send votes with node listening channel
  Redis ->> ControlNode: Count votes
  ControlNode ->> Redis: Inform nodes that went into next circle
  Redis ->> FreeNodes: Get voting results
  FreeNodes ->> FreeNodes: Switch to committee member if selected

  ```
