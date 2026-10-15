# Agents with LangGraph

LangGraph builds agents as graphs. Each node is a function that reads the shared state and returns an update to it. Edges decide which node runs next; a conditional edge picks the next node from the current state, which is how an agent branches, loops, or decides it is finished.

## State and checkpoints

The graph's state is a typed dictionary. With a checkpointer attached, LangGraph saves the state after every step, keyed by a thread ID. That gives an agent memory across turns, lets a long run resume after a failure, and makes it possible to inspect or rewind any earlier step.

## Human in the loop

A node can call interrupt to pause the graph and wait for a person, for example to approve a tool call before it runs. The paused state is stored by the checkpointer, and the run continues when it is resumed with the person's answer.

## Streaming

A compiled graph can stream several kinds of output while it runs: state updates after each node, tokens from chat models, and custom events that a node writes itself. A web app usually forwards these to the browser as server-sent events, so the user sees sources and the answer appear as they are produced.
