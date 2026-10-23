# 13 · JS/TS 版本速查

> **本章目标**：面向前端 / 全栈同学，把前面用 Python 学到的概念快速"翻译"成 TypeScript 写法，并能在 Node.js / Next.js 项目中跑起来。
>
> **前置知识**：熟悉 TypeScript、`async/await`、npm；读过本教程 02~10 章（至少理解概念）。

---

## 一、先搞清楚：JS 版和 Python 版是什么关系

- LangChain 和 LangGraph 都有**官方维护**的 Python 版和 JS/TS 版，**概念完全一致**，API 设计也高度对应。
- Python 版是"一等公民"：新功能通常先发 Python，第三方集成（文档加载器、向量库等）也更多。
- JS 版的优势：和前端、Node 后端、Next.js 同一种语言，类型提示好，可以部署到 Vercel 等平台。

> 💡 **通俗理解**：同一套设计图纸，分别用两种材料盖的房子。房间布局一样，只是门把手的样子略有不同（命名风格不同）。

**命名规律**（记住这几条，大部分 API 能自己推出来）：

| 规律 | Python | TypeScript |
|---|---|---|
| 函数/方法命名 | `snake_case` | `camelCase` |
| 创建 Agent | `create_agent` | `createAgent` |
| 初始化模型 | `init_chat_model` | `initChatModel`（返回 Promise，需要 `await`） |
| 参数 | `system_prompt=...` | `{ systemPrompt: ... }` |
| 流式模式参数 | `stream_mode="messages"` | `{ streamMode: "messages" }` |
| 调用方法 | `invoke` / `ainvoke`（异步） | `invoke`（本身就是异步，返回 Promise） |
| Schema 定义 | Pydantic / TypedDict | **zod** |
| 中间件 | `SummarizationMiddleware(...)` 类 | `summarizationMiddleware(...)` 工厂函数 |

> 📖 **术语**：**zod**：TypeScript 生态最流行的运行时校验库，用来定义数据结构并在运行时校验。在 LangChain.js 中，它承担了 Python 版里 Pydantic 的角色：定义工具参数、结构化输出、图的状态。

> ⚠️ **注意**：本章写法基于 LangChain.js 1.x / LangGraph.js 1.x。JS 版近年 API 变化较大（例如状态定义从 `Annotation` 演进到 `StateSchema`），网上旧教程请谨慎参考，以 [官方 JS 文档](https://docs.langchain.com/oss/javascript/langchain/overview) 为准。

---

## 二、环境准备

**要求**：Node.js ≥ 22（官方要求），推荐使用 pnpm。

```bash
mkdir lc-ts-lab && cd lc-ts-lab
pnpm init
pnpm add langchain @langchain/core @langchain/langgraph @langchain/deepseek zod dotenv
pnpm add -D typescript tsx @types/node
```

| 包 | 对应 Python | 作用 |
|---|---|---|
| `langchain` | `langchain` | `createAgent`、`tool`、`initChatModel`、中间件 |
| `@langchain/core` | `langchain-core` | 消息、提示词模板、Runnable 等核心抽象 |
| `@langchain/langgraph` | `langgraph` | `StateGraph`、checkpointer、`interrupt` 等 |
| `@langchain/deepseek` | `langchain-deepseek` | DeepSeek 接入（学习阶段默认） |
| `@langchain/openai` | `langchain-openai` | OpenAI 及 OpenAI 兼容接口（通义千问等） |
| `zod` | `pydantic` | Schema 定义 |

> 📖 **术语**：**tsx**：直接运行 `.ts` 文件的工具，无需先编译。`npx tsx hello.ts` 即可运行。

`.env`：

```bash
DEEPSEEK_API_KEY=sk-xxxxxxxx
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_xxxxxxxx
```

`package.json` 里加上 `"type": "module"`，以便使用 ESM 和顶层 `await`。

> ⚠️ **注意**：与 Python 版相同，DeepSeek 默认开启思考模式，而带工具调用时需要回传 `reasoning_content`，LangChain 不会自动处理。因此 JS 示例统一通过 `modelKwargs: { thinking: { type: "disabled" } }` 关闭思考模式（原理见 [01 章 4.1 节](./01-环境准备.md#41-deepseek本教程默认)）。`createAgent` 的 `model` 也要传模型实例，不能只传字符串。

---

## 三、Python ↔ TypeScript 对照总表

| 功能 | Python | TypeScript |
|---|---|---|
| 初始化模型 | `init_chat_model("deepseek:deepseek-flash", extra_body={...})` | `await initChatModel("deepseek:deepseek-flash", { modelKwargs: {...} })` |
| 直接实例化 | `ChatDeepSeek(model=..., extra_body={...})` | `new ChatDeepSeek({ model: ..., modelKwargs: {...} })` |
| 关闭思考模式 | `extra_body={"thinking": {"type": "disabled"}}` | `modelKwargs: { thinking: { type: "disabled" } }` |
| 调用 | `model.invoke("hi")` | `await model.invoke("hi")` |
| 流式 | `for chunk in model.stream(...)` | `for await (const chunk of await model.stream(...))` |
| 批量 | `model.batch([...])` | `await model.batch([...])` |
| 消息 | `HumanMessage("hi")` | `new HumanMessage("hi")` |
| 提示词模板 | `ChatPromptTemplate.from_messages([...])` | `ChatPromptTemplate.fromMessages([...])` |
| 管道 | `prompt \| model \| parser` | `prompt.pipe(model).pipe(parser)` |
| 结构化输出 | `model.with_structured_output(PydanticModel)` | `model.withStructuredOutput(zodSchema)` |
| 定义工具 | `@tool` 装饰器 | `tool(fn, { name, description, schema })` |
| 绑定工具 | `model.bind_tools([...])` | `model.bindTools([...])` |
| 创建 Agent | `create_agent(model=..., tools=..., system_prompt=...)` | `createAgent({ model, tools, systemPrompt })` |
| Agent 结构化结果 | `result["structured_response"]` | `result.structuredResponse` |
| 内存 checkpointer | `InMemorySaver()` | `new MemorySaver()` |
| 会话 ID | `config={"configurable": {"thread_id": "1"}}` | `{ configurable: { thread_id: "1" } }`（注意 `thread_id` 仍是下划线） |
| 图状态 | `TypedDict` / `MessagesState` | `new StateSchema({...})` + `MessagesValue` |
| 构建图 | `StateGraph(State)` | `new StateGraph(State)` |
| 加节点/边 | `add_node` / `add_edge` / `add_conditional_edges` | `addNode` / `addEdge` / `addConditionalEdges` |
| 中断 | `interrupt(value)` | `interrupt(value)` |
| 恢复 | `Command(resume=...)` | `new Command({ resume: ... })` |
| 并行分发 | `Send("node", {...})` | `new Send("node", {...})` |

---

## 四、模型与消息

```ts
// 01-model.ts
import "dotenv/config";
import { initChatModel } from "langchain";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

const model = await initChatModel("deepseek:deepseek-flash", { modelKwargs: { thinking: { type: "disabled" } },
  temperature: 0.3,
  maxTokens: 1000,
});

// 1. 最简单的调用
const res = await model.invoke("用一句话介绍 LangChain");
console.log(res.text);

// 2. 带系统消息的多轮消息
const res2 = await model.invoke([
  new SystemMessage("你是一名资深前端工程师，回答简洁"),
  new HumanMessage("React 和 Vue 的核心区别是什么？"),
]);
console.log(res2.text);

// 3. 也可以直接用对象形式的消息
const res3 = await model.invoke([
  { role: "system", content: "你只用 emoji 回答" },
  { role: "user", content: "今天心情怎么样？" },
]);
console.log(res3.text);

// 4. 流式输出
const stream = await model.stream("写一首关于程序员的四行小诗");
for await (const chunk of stream) {
  process.stdout.write(chunk.text);
}
```

运行：`npx tsx 01-model.ts`

> 💡 `.text` 会把消息内容统一转成字符串；`.content` 可能是字符串，也可能是内容块数组（多模态、工具调用时）。展示文本时优先用 `.text`。

**使用国内模型（OpenAI 兼容接口）**：

```ts
import { ChatOpenAI } from "@langchain/openai";

const qwen = new ChatOpenAI({
  model: "qwen-plus",
  apiKey: process.env.DASHSCOPE_API_KEY,
  configuration: {
    baseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1",
  },
});
```

> ⚠️ **注意**：JS 版的 `ChatOpenAI` 通过 `configuration.baseURL` 设置地址（大写 `URL`），和 Python 的 `base_url` 不同。

---

## 五、提示词模板与结构化输出

```ts
// 02-prompt.ts
import "dotenv/config";
import { initChatModel } from "langchain";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import * as z from "zod";

const model = await initChatModel("deepseek:deepseek-flash", { modelKwargs: { thinking: { type: "disabled" } }, temperature: 0 });

// 提示词模板 + 管道
const prompt = ChatPromptTemplate.fromMessages([
  ["system", "你是一名翻译，把用户输入翻译成{language}，只输出译文"],
  ["human", "{text}"],
]);

const chain = prompt.pipe(model);
const res = await chain.invoke({ language: "英文", text: "今天天气真好" });
console.log(res.text);

// 结构化输出：用 zod 定义你想要的数据结构
const Resume = z.object({
  name: z.string().describe("姓名"),
  years: z.number().describe("工作年限"),
  skills: z.array(z.string()).describe("技能列表"),
});

const extractor = model.withStructuredOutput(Resume);

const info = await extractor.invoke(
  "我叫张三，做了 5 年前端，熟悉 Vue、React、TypeScript，最近在学 Rust。",
);
console.log(info); // { name: "张三", years: 5, skills: ["Vue", "React", "TypeScript", "Rust"] }
// info 的类型会被自动推断为 z.infer<typeof Resume>
```

> 🎯 **应用场景**：表单智能填充、简历解析、从客服对话中抽取订单号和问题类型——前端拿到的直接是类型安全的对象。

---

## 六、工具调用

```ts
// 03-tools.ts
import "dotenv/config";
import { initChatModel, tool } from "langchain";
import * as z from "zod";

const getWeather = tool(
  async ({ city }) => {
    // 实际项目中这里调用天气 API
    return `${city}今天晴，25°C`;
  },
  {
    name: "get_weather",
    description: "查询指定城市的天气",
    schema: z.object({
      city: z.string().describe("城市名，如：北京"),
    }),
  },
);

const model = await initChatModel("deepseek:deepseek-flash", { modelKwargs: { thinking: { type: "disabled" } } });
const modelWithTools = model.bindTools([getWeather]);

const aiMsg = await modelWithTools.invoke("上海今天天气怎么样？");
console.log(aiMsg.tool_calls);
// [{ name: "get_weather", args: { city: "上海" }, id: "toolu_...", type: "tool_call" }]

// 手动执行工具，并把结果回传给模型（完整循环见第 05 章；实际项目直接用 createAgent）
const toolMsg = await getWeather.invoke(aiMsg.tool_calls![0]);
const final = await modelWithTools.invoke([
  { role: "user", content: "上海今天天气怎么样？" },
  aiMsg,
  toolMsg,
]);
console.log(final.text);
```

> 💡 和 Python 的 `@tool` 用函数 docstring 作为描述不同，JS 中 `name`、`description`、`schema` 需要显式写出来。

---

## 七、Agent：createAgent

```ts
// 04-agent.ts
import "dotenv/config";
import { createAgent, tool } from "langchain";
import { MemorySaver } from "@langchain/langgraph";
import * as z from "zod";
import { ChatDeepSeek } from "@langchain/deepseek";

const getWeather = tool(async ({ city }) => `${city}今天晴，25°C`, {
  name: "get_weather",
  description: "查询指定城市的天气",
  schema: z.object({ city: z.string() }),
});

const agent = createAgent({
  model: new ChatDeepSeek({ model: "deepseek-flash", modelKwargs: { thinking: { type: "disabled" } } }),
  tools: [getWeather],
  systemPrompt: "你是一个乐于助人的助手，用中文回答。",
  checkpointer: new MemorySaver(), // 短期记忆
});

const config = { configurable: { thread_id: "user-1" } };

const r1 = await agent.invoke(
  { messages: [{ role: "user", content: "我叫小明，我在杭州" }] },
  config,
);
console.log(r1.messages.at(-1)?.text);

const r2 = await agent.invoke(
  { messages: [{ role: "user", content: "我这边今天适合出门吗？" }] },
  config, // 同一个 thread_id，Agent 记得你在杭州
);
console.log(r2.messages.at(-1)?.text);
```

**结构化输出**：

```ts
import { ChatDeepSeek } from "@langchain/deepseek";
const WeatherReport = z.object({
  city: z.string(),
  suitableForOutdoor: z.boolean(),
  advice: z.string(),
});

const reporter = createAgent({
  model: new ChatDeepSeek({ model: "deepseek-flash", modelKwargs: { thinking: { type: "disabled" } } }),
  tools: [getWeather],
  responseFormat: WeatherReport,
});

const result = await reporter.invoke({
  messages: [{ role: "user", content: "杭州今天适合出门吗？" }],
});
console.log(result.structuredResponse); // { city: "杭州", suitableForOutdoor: true, advice: "..." }
```

**中间件**（与 Python 对应，命名为小驼峰工厂函数）：

```ts
import { createAgent, summarizationMiddleware, humanInTheLoopMiddleware } from "langchain";
import { MemorySaver } from "@langchain/langgraph";
import { ChatDeepSeek } from "@langchain/deepseek";

const agent = createAgent({
  model: new ChatDeepSeek({ model: "deepseek-flash", modelKwargs: { thinking: { type: "disabled" } } }),
  tools: [sendEmail, readEmail],
  checkpointer: new MemorySaver(),
  middleware: [
    summarizationMiddleware({
      model: new ChatDeepSeek({ model: "deepseek-flash", modelKwargs: { thinking: { type: "disabled" } } }),
      trigger: { tokens: 4000 },
    }),
    humanInTheLoopMiddleware({
      interruptOn: { send_email: true, read_email: false },
    }),
  ],
});
```

> ⚠️ **注意**：中间件的参数格式在 JS 和 Python 之间并非完全一致（例如 Python 的 `trigger=("tokens", 4000)` 元组在 JS 中是对象），使用时查阅 [JS 中间件文档](https://docs.langchain.com/oss/javascript/langchain/middleware/built-in)。

---

## 八、LangGraph：StateGraph

### 8.1 定义状态、节点、边

```ts
// 05-graph.ts
import "dotenv/config";
import { StateGraph, StateSchema, MessagesValue, START, END } from "@langchain/langgraph";
import { initChatModel } from "langchain";
import * as z from "zod";

const model = await initChatModel("deepseek:deepseek-flash", { modelKwargs: { thinking: { type: "disabled" } }, temperature: 0 });

// 状态 = 图里所有节点共享的数据
const State = new StateSchema({
  messages: MessagesValue,                       // 等价于 Python 的 Annotated[list, add_messages]
  category: z.string().default(""),              // 普通字段：新值覆盖旧值
});

// 节点 = 一个函数：接收状态，返回要更新的部分
const classify = async (state: typeof State.State) => {
  const res = await model.invoke([
    { role: "system", content: "判断用户问题类型，只回答：技术 或 闲聊" },
    ...state.messages,
  ]);
  return { category: res.text.includes("技术") ? "技术" : "闲聊" };
};

const techAnswer = async (state: typeof State.State) => {
  const res = await model.invoke([
    { role: "system", content: "你是资深工程师，回答专业、给出代码示例" },
    ...state.messages,
  ]);
  return { messages: [res] };
};

const chatAnswer = async (state: typeof State.State) => {
  const res = await model.invoke([
    { role: "system", content: "你是一个轻松幽默的聊天伙伴" },
    ...state.messages,
  ]);
  return { messages: [res] };
};

// 条件边的路由函数：返回下一个节点名
const route = (state: typeof State.State) => (state.category === "技术" ? "techAnswer" : "chatAnswer");

const graph = new StateGraph(State)
  .addNode("classify", classify)
  .addNode("techAnswer", techAnswer)
  .addNode("chatAnswer", chatAnswer)
  .addEdge(START, "classify")
  .addConditionalEdges("classify", route, ["techAnswer", "chatAnswer"])
  .addEdge("techAnswer", END)
  .addEdge("chatAnswer", END)
  .compile();

const result = await graph.invoke({
  messages: [{ role: "user", content: "TypeScript 的 infer 关键字怎么用？" }],
});
console.log(result.category, result.messages.at(-1)?.text);
```

**状态字段写法对照**：

| 需求 | Python | TypeScript |
|---|---|---|
| 消息列表（追加合并） | `Annotated[list, add_messages]` 或继承 `MessagesState` | `messages: MessagesValue` |
| 普通字段（覆盖） | `category: str` | `category: z.string()` |
| 带默认值 | 在 `invoke` 输入中提供 | `z.number().default(0)` |
| 自定义合并规则 | `Annotated[list, operator.add]` | `new ReducedValue(z.array(z.string()).default(() => []), { reducer: (a, b) => a.concat(b) })` |

> 💡 你可能在旧教程中见到 `Annotation.Root({...})` 和 `MessagesAnnotation` 的写法，那是早期的状态定义方式。1.x 文档推荐使用 `StateSchema` + zod。

### 8.2 持久化与人机协同（interrupt / Command）

```ts
// 06-interrupt.ts
import { StateGraph, StateSchema, START, END, MemorySaver, interrupt, Command } from "@langchain/langgraph";
import * as z from "zod";

const State = new StateSchema({
  amount: z.number(),
  approved: z.boolean().default(false),
});

const approval = (state: typeof State.State) => {
  // 暂停执行，把信息交给人审核；恢复时 interrupt() 的返回值就是人给的答复
  const answer = interrupt({ question: `确认转账 ${state.amount} 元吗？` });
  return { approved: answer === "yes" };
};

const transfer = (state: typeof State.State) => {
  console.log(state.approved ? `✅ 已转账 ${state.amount} 元` : "❌ 已取消");
  return {};
};

const graph = new StateGraph(State)
  .addNode("approval", approval)
  .addNode("transfer", transfer)
  .addEdge(START, "approval")
  .addEdge("approval", "transfer")
  .addEdge("transfer", END)
  .compile({ checkpointer: new MemorySaver() }); // 中断必须配 checkpointer

const config = { configurable: { thread_id: "tx-001" } };

const first = await graph.invoke({ amount: 500 }, config);
console.log(first.__interrupt__); // [{ value: { question: "确认转账 500 元吗？" }, ... }]

// 模拟用户在前端点了"同意"，用同一个 thread_id 恢复
await graph.invoke(new Command({ resume: "yes" }), config);
```

> ⚠️ **注意**：和 Python 一样，恢复时**整个节点会从头重新执行**，`interrupt()` 之前的代码会再跑一遍，不要在它前面放有副作用的操作。详见第 09 章。

生产环境的持久化：

```bash
pnpm add @langchain/langgraph-checkpoint-postgres
```

```ts
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";

const checkpointer = PostgresSaver.fromConnString(process.env.DATABASE_URL!);
await checkpointer.setup(); // 首次运行建表
```

### 8.3 流式输出

```ts
const stream = await agent.stream(
  { messages: [{ role: "user", content: "杭州天气怎么样？" }] },
  { streamMode: "messages", configurable: { thread_id: "user-1" } },
);

for await (const [chunk, metadata] of stream) {
  if (metadata.langgraph_node === "model" && chunk.text) {
    process.stdout.write(chunk.text); // 逐字输出
  }
}
```

`streamMode` 的取值与 Python 一致：`"values"`（每步完整状态）、`"updates"`（每步增量）、`"messages"`（模型逐 token）、`"custom"`（自定义事件），也可以传数组同时订阅多种。

---

## 九、在 Next.js 中使用

### 9.1 Route Handler 流式返回

目录结构（App Router）：

```
app/
├── api/chat/route.ts     # 后端接口
└── page.tsx              # 聊天页面
lib/
└── agent.ts              # Agent 定义
```

`lib/agent.ts`：

```ts
import { createAgent, tool } from "langchain";
import { MemorySaver } from "@langchain/langgraph";
import * as z from "zod";
import { ChatDeepSeek } from "@langchain/deepseek";

const getWeather = tool(async ({ city }) => `${city}今天晴，25°C`, {
  name: "get_weather",
  description: "查询指定城市的天气",
  schema: z.object({ city: z.string() }),
});

// 模块级单例：避免每个请求都重新创建
// 注意：MemorySaver 只在单进程内有效，Serverless/多实例部署必须换成 PostgresSaver
export const agent = createAgent({
  model: new ChatDeepSeek({ model: "deepseek-flash", modelKwargs: { thinking: { type: "disabled" } } }),
  tools: [getWeather],
  systemPrompt: "你是一个乐于助人的助手，用中文回答。",
  checkpointer: new MemorySaver(),
});
```

`app/api/chat/route.ts`：

```ts
import { agent } from "@/lib/agent";

export const runtime = "nodejs"; // LangChain 部分依赖需要 Node 运行时

export async function POST(req: Request) {
  const { message, threadId } = await req.json();
  const thread_id = threadId ?? crypto.randomUUID();

  const encoder = new TextEncoder();
  const sse = (event: string, data: unknown) =>
    encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);

  const readable = new ReadableStream({
    async start(controller) {
      controller.enqueue(sse("start", { thread_id }));
      try {
        const stream = await agent.stream(
          { messages: [{ role: "user", content: message }] },
          {
            streamMode: "messages",
            configurable: { thread_id },
            signal: req.signal, // 前端中断请求时，同步取消 Agent 执行
          },
        );
        for await (const [chunk, metadata] of stream) {
          if (metadata.langgraph_node === "model" && chunk.text) {
            controller.enqueue(sse("token", { text: chunk.text }));
          }
        }
        controller.enqueue(sse("end", {}));
      } catch (err) {
        controller.enqueue(sse("error", { message: "服务暂时不可用" }));
      } finally {
        controller.close();
      }
    },
  });

  return new Response(readable, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}
```

### 9.2 前端页面

```tsx
// app/page.tsx
"use client";

import { useRef, useState } from "react";

type Msg = { role: "user" | "assistant"; content: string };

export default function ChatPage() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const threadId = useRef<string | undefined>(undefined);
  const abortRef = useRef<AbortController | null>(null);

  async function send() {
    if (!input.trim()) return;
    const userText = input;
    setInput("");
    setLoading(true);
    setMessages((prev) => [...prev, { role: "user", content: userText }, { role: "assistant", content: "" }]);

    abortRef.current = new AbortController();
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: userText, threadId: threadId.current }),
      signal: abortRef.current.signal,
    });

    const reader = res.body!.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = "";
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += value;
        const parts = buffer.split("\n\n");
        buffer = parts.pop()!;
        for (const part of parts) {
          const event = part.match(/^event: (.*)$/m)?.[1];
          const data = JSON.parse(part.match(/^data: (.*)$/m)?.[1] ?? "{}");
          if (event === "start") threadId.current = data.thread_id;
          if (event === "token") {
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              next[next.length - 1] = { ...last, content: last.content + data.text };
              return next;
            });
          }
        }
      }
    } catch {
      // 用户点击了停止
    } finally {
      setLoading(false);
    }
  }

  return (
    <main style={{ maxWidth: 720, margin: "40px auto" }}>
      {messages.map((m, i) => (
        <p key={i}>
          <b>{m.role === "user" ? "我" : "AI"}：</b>
          {m.content}
        </p>
      ))}
      <input value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && send()} />
      <button onClick={send} disabled={loading}>发送</button>
      {loading && <button onClick={() => abortRef.current?.abort()}>停止</button>}
    </main>
  );
}
```

> ⚠️ **注意**：
> - API Key 只能出现在服务端代码（Route Handler、`lib/`），**绝不能**用 `NEXT_PUBLIC_` 前缀暴露给浏览器。
> - Vercel 等 Serverless 平台上的函数有最长执行时间限制，长时间运行的 Agent 需要调整 `maxDuration` 或改用常驻服务。
> - 实例之间不共享内存，多轮对话必须使用数据库 checkpointer。

### 9.3 对接 LangGraph Server：useStream

如果后端使用 `langgraph dev` / LangGraph Server（第 12 章），前端可以直接使用官方 Hook，省去手写 SSE 解析和消息状态管理：

```bash
pnpm add @langchain/react   # Vue: @langchain/vue  Svelte: @langchain/svelte  Angular: @langchain/angular
```

```tsx
import { useStream } from "@langchain/react";

export function Chat() {
  const stream = useStream({
    apiUrl: "http://localhost:2024",
    assistantId: "agent",
  });

  return (
    <div>
      {stream.messages.map((msg) => (
        <div key={msg.id}>{String(msg.content)}</div>
      ))}
    </div>
  );
}
```

它还提供 `submit`（发送消息）、`interrupt`（人机协同中断信息）、`toolCalls` 等能力。具体 API 以 [官方前端文档](https://docs.langchain.com/oss/javascript/langchain/streaming/frontend) 为准。

---

## 十、与 Vercel AI SDK 的选型对比

> 📖 **术语**：**Vercel AI SDK**（`ai` 包）：Vercel 推出的 TypeScript AI 开发工具包，主打统一的模型调用接口、流式 UI（`useChat` Hook）以及工具调用，和 Next.js 结合非常紧密。

| 维度 | LangChain.js / LangGraph.js | Vercel AI SDK |
|---|---|---|
| **定位** | 完整的 LLM 应用框架 + 流程编排引擎 | 轻量的模型调用 + 流式 UI 工具包 |
| **上手难度** | 中等，概念较多 | 低，前端同学几乎零门槛 |
| **前端流式 UI** | 需要自己处理，或使用 `useStream`（需要 LangGraph Server） | `useChat` 开箱即用，体验极佳 |
| **工具调用 / 简单 Agent** | `createAgent` | `generateText` / `streamText` + `tools`，以及 Agent 封装 |
| **复杂流程（分支、循环、并行）** | LangGraph 专长 | 需要自己写代码编排 |
| **持久化、断点续跑、人机协同** | checkpointer + `interrupt` 原生支持 | 需要自己实现 |
| **RAG 生态** | 文档加载器、切分器、向量库集成丰富 | 较少，通常自己组合 |
| **观测与评估** | LangSmith 深度集成 | 支持 OpenTelemetry，可接入多种平台 |
| **与 Python 团队协作** | 概念一致，团队可以共享设计 | 仅 TS |

**选型建议**：

- **页面里加一个聊天框、做总结/翻译/简单工具调用** → Vercel AI SDK，最省事。
- **RAG 知识库、需要多步骤、需要人工审批和断点续跑、多 Agent** → LangChain / LangGraph。
- **两者结合**：后端用 LangGraph 编排复杂逻辑，前端用 AI SDK 的 `useChat` 渲染。AI SDK 提供了 LangChain 适配器（`@ai-sdk/langchain`），可以把 LangChain 的流转换成 AI SDK 的 UI 消息流，具体用法以 [AI SDK 文档](https://ai-sdk.dev) 为准。
- **团队有 Python AI 工程师**：AI 核心逻辑用 Python 写成服务，前端和 BFF 层用 TS 调用。这是中大型项目最常见的架构。

---

## 十一、JS 版常见坑

| 问题 | 原因与解决 |
|---|---|
| `Cannot use import statement outside a module` / 顶层 `await` 报错 | `package.json` 加 `"type": "module"`，或使用 `tsx` 运行 |
| `initChatModel` 返回的是 Promise | 必须 `await initChatModel(...)` |
| 结构化输出类型推断不正确 | 检查 zod 版本，与 `@langchain/core` 要求的版本保持一致 |
| 包版本冲突、`instanceof` 判断失效 | 项目中存在多份 `@langchain/core`。确保所有 `@langchain/*` 包版本兼容，必要时在 `package.json` 里用 `overrides`/`resolutions` 固定 `@langchain/core` 版本 |
| Edge Runtime 下报错 | 部分集成依赖 Node API，Route Handler 中设置 `export const runtime = "nodejs"` |
| 部署后多轮对话"失忆" | Serverless 多实例不共享 `MemorySaver`，换成 Postgres 等持久化 checkpointer |
| 照着 Python 教程写，参数不生效 | JS 参数是 camelCase（`systemPrompt`、`streamMode`、`maxTokens`），但 `configurable.thread_id` 保持下划线 |

---

## 本章小结

- JS 版与 Python 版**概念一致**，主要差异是：camelCase 命名、`await`、用 **zod** 代替 Pydantic、`tool()` 需要显式写 `name`/`description`/`schema`。
- 核心 API：`initChatModel`、`withStructuredOutput`、`tool`、`createAgent`、`StateGraph` + `StateSchema`、`MemorySaver`、`interrupt` + `Command`、`stream({ streamMode })`。
- 在 Next.js 中：Route Handler + `ReadableStream` 输出 SSE，前端 `fetch` 读取流；API Key 只放服务端；生产环境使用数据库 checkpointer。
- 简单聊天 UI 选 Vercel AI SDK；复杂流程、RAG、人机协同选 LangChain/LangGraph；两者可以结合使用。

## 练习

1. 把第 07 章的 Python Agent 用 TypeScript 重写一遍，保持相同的工具和系统提示词。
2. 用 `StateSchema` 实现第 08 章的"问题分类 → 分支回答"图。
3. 在 Next.js 项目中实现 9.1 和 9.2 的流式聊天页面，并加上"停止生成"和 Markdown 渲染。
4. 进阶：用 `langgraph dev`（JS 版同样支持 `langgraph.json`）启动服务，再用 `useStream` 实现一个带人工审批按钮的前端页面。

---

**上一章** ← [12 调试评估与生产部署](./12-调试评估与生产部署.md) ｜ **下一章** → [14 FAQ 与学习资源](./14-FAQ与学习资源.md)
