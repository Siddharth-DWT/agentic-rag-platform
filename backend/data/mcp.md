# Model Context Protocol

The Model Context Protocol (MCP) is an open protocol for connecting AI applications to tools and data. An MCP server exposes tools, resources and prompts. An MCP client, built into an application such as an AI assistant or an IDE, discovers what a server offers and calls it on the model's behalf.

MCP messages are JSON-RPC 2.0. A client starts a session by sending an initialize request; the two sides agree on a protocol version and exchange capabilities, and the client can then list and call tools.

## Transports

A local MCP server runs as a child process and talks over standard input and output (stdio). A remote MCP server is reached over HTTP using the Streamable HTTP transport: the client POSTs JSON-RPC messages to one endpoint, and the server answers with JSON or with a server-sent event stream when it needs to send several messages.

## Authorization

Remote MCP servers use OAuth 2.1. The MCP server acts as a resource server: it publishes protected resource metadata that names its authorization server, and it accepts requests only with a valid bearer access token issued for it. Clients discover the authorization server from that metadata, register, and obtain a token with the authorization code flow and PKCE. The server must check that each token was issued for its own resource, using the token's audience, so a token meant for one server cannot be replayed against another.
