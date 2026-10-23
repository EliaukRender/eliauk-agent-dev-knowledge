# 09 · LangGraph 持久化、记忆与人机协同

> **本章目标**：掌握 LangGraph 的"超能力"：用 Checkpointer 实现多轮对话、断点续跑和"时间旅行"；管理短期记忆的长度；用 Store 实现跨会话的长期记忆；用 `interrupt` 在任意步骤暂停，等人审批、修改或补充信息后再继续。
>
> **前置知识**：[08 LangGraph 核心概念](./08-LangGraph核心概念.md)。

---

## 一、全景：四种能力，两套机制

```mermaid
graph TB
    subgraph CP[Checkpointer · 按 thread_id 保存每一步的状态]
        A[多轮对话<br/>短期记忆]
        B[断点续跑<br/>容错恢复]
        C[时间旅行<br/>回溯 / 分叉]
        D[人机协同<br/>暂停 / 恢复]
    end
    subgraph ST[Store · 跨 thread 的键值存储]
        E[长期记忆<br/>用户偏好 / 知识沉淀]
    end
```

| 机制 | 保存什么 | 作用范围 | 类比 |
|---|---|---|---|
| **Checkpointer** | 图在每一步执行后的**完整状态快照** | 单个会话（`thread_id`） | 游戏的自动存档 |
| **Store** | 你主动写入的任意数据 | 跨会话、跨用户（按 namespace 组织） | 游戏账号里的永久成就和设置 |

---

## 二、Checkpointer：给图加上"自动存档"

### 2.1 基本概念

> 📖 **术语**：**Checkpoint（检查点）**——图执行过程中某一时刻的状态快照，包含当时的所有状态值、下一步要执行的节点等信息。

> 📖 **术语**：**Checkpointer（检查点保存器）**——负责保存和读取 Checkpoint 的组件。编译图时传入后，**每执行完一个步骤（super-step）就自动存一次档**。

> 📖 **术语**：**Thread（线程 / 会话）**——一串 Checkpoint 组成的执行历史，用 `thread_id` 标识。注意它和操作系统的"线程"无关，理解为"一个会话"或"一个任务实例"即可。

> 💡 **通俗理解**：Checkpointer 就像游戏的自动存档。每过一关（执行一个节点）就存一次档；`thread_id` 是存档位，"存档位 1"是你的进度，"存档位 2"是你朋友的进度，互不影响。随时可以读档继续玩，甚至可以回到某一关重新选择。

### 2.2 多轮对话

```python
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

load_dotenv()

model = init_chat_model("deepseek:deepseek-flash", extra_body={"thinking": {"type": "disabled"}})


def chatbot(state: MessagesState) -> dict:
    return {"messages": [model.invoke(state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node(chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile(checkpointer=InMemorySaver())  # 关键：编译时挂上 checkpointer

config = {"configurable": {"thread_id": "chat-001"}}

graph.invoke({"messages": [{"role": "user", "content": "我在学 LangGraph，我是前端工程师"}]}, config)
result = graph.invoke({"messages": [{"role": "user", "content": "根据我的背景，推荐一个练手项目"}]}, config)
print(result["messages"][-1].content)  # 会结合"前端工程师"给出建议

# 查看当前状态
snapshot = graph.get_state(config)
print(f"当前共 {len(snapshot.values['messages'])} 条消息，下一步：{snapshot.next}")
```

`get_state` 返回的 `StateSnapshot` 常用字段：

| 字段 | 含义 |
|---|---|
| `values` | 当前状态值 |
| `next` | 接下来要执行的节点（空元组表示已执行完毕） |
| `config` | 这个 Checkpoint 的配置（含 `checkpoint_id`），可用于回溯 |
| `metadata` | 元数据，如是第几步、由哪个节点写入 |
| `tasks` | 待执行的任务，包含中断信息 |

### 2.3 Checkpointer 选型

| 实现 | 安装 | 适合 |
|---|---|---|
| `InMemorySaver` | 内置 | 学习、测试。**程序重启数据就没了** |
| `SqliteSaver` | `uv add langgraph-checkpoint-sqlite` | 本地开发、单机小应用、桌面应用 |
| `PostgresSaver` | `uv add langgraph-checkpoint-postgres "psycopg[binary,pool]"` | **生产环境首选** |
| 其他 | Redis、MongoDB 等社区实现 | 按团队技术栈选择 |

**SqliteSaver**：

```python
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

graph = builder.compile(checkpointer=checkpointer)
# 现在关掉程序再打开，用同一个 thread_id 依然能接着聊
```

**PostgresSaver**：

```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()  # 首次使用时建表，只需执行一次
    graph = builder.compile(checkpointer=checkpointer)
    graph.invoke({"messages": [{"role": "user", "content": "你好"}]}, {"configurable": {"thread_id": "1"}})
```

> 💡 异步应用（如 FastAPI）使用对应的异步版本：`AsyncSqliteSaver`、`AsyncPostgresSaver`，配合 `ainvoke` / `astream`。

> ⚠️ **注意**：`create_agent` 的 `checkpointer` 参数和这里是同一个东西，第 07 章学到的多轮记忆就是基于本节的机制实现的。

### 2.4 断点续跑与容错

因为每一步都存了档，**如果某个节点执行到一半失败了（比如 API 超时），修好问题后用同一个 `thread_id` 重新调用，图会从失败的那一步继续，而不是从头开始**。已成功的节点不会重复执行。

```python
config = {"configurable": {"thread_id": "report-job-42"}}
try:
    graph.invoke({"topic": "2026 新能源市场"}, config)
except Exception as e:
    print(f"执行失败：{e}")

# 修复问题后（比如网络恢复），传入 None 表示"从上次的位置继续"
graph.invoke(None, config)
```

> 🎯 **应用场景**：长时间运行的任务（批量生成报告、多步骤数据处理、深度调研 Agent），中途失败可以从断点恢复，省时间也省 Token 费用。

调用时还可以用 `durability` 参数控制存档的时机，在性能和可靠性之间取舍：

| 取值 | 行为 | 适合 |
|---|---|---|
| `"exit"` | 只在图执行结束（或中断、出错）时保存 | 追求性能，不需要中间步骤恢复 |
| `"async"`（默认） | 异步保存，下一步执行的同时写入 | 大多数场景 |
| `"sync"` | 每一步保存完成后才执行下一步 | 对可靠性要求最高的任务 |

```python
graph.invoke(inputs, config, durability="sync")
```

---

## 三、时间旅行：回溯与分叉

> 📖 **术语**：**Time Travel（时间旅行）**——查看图的历史执行状态，从过去的某个 Checkpoint 重新执行（重放），或修改历史状态后走出一条新路径（分叉）。

> 💡 **通俗理解**：读取旧存档。你可以回到第 3 关重新打（重放），也可以回到第 3 关时先改一下装备再打（分叉），原来的存档依然保留。

### 3.1 查看历史

```python
from typing import TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

load_dotenv()

model = init_chat_model("deepseek:deepseek-flash", extra_body={"thinking": {"type": "disabled"}})


class State(TypedDict):
    topic: str
    outline: str
    article: str


def make_outline(state: State) -> dict:
    return {"outline": model.invoke(f"为《{state['topic']}》列一个 3 点提纲，每点一句话").content}


def write_article(state: State) -> dict:
    return {"article": model.invoke(f"按提纲写一篇 200 字短文：\n{state['outline']}").content}


builder = StateGraph(State)
builder.add_node(make_outline)
builder.add_node(write_article)
builder.add_edge(START, "make_outline")
builder.add_edge("make_outline", "write_article")
builder.add_edge("write_article", END)
graph = builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "article-1"}}
graph.invoke({"topic": "远程办公的利与弊"}, config)

# 历史按时间倒序返回（最新的在前）
history = list(graph.get_state_history(config))
for snap in history:
    print(f"step={snap.metadata['step']:>2}  next={snap.next}  keys={list(snap.values.keys())}")
```

输出类似：

```
step= 2  next=()                   keys=['topic', 'outline', 'article']
step= 1  next=('write_article',)   keys=['topic', 'outline']
step= 0  next=('make_outline',)    keys=['topic']
step=-1  next=('__start__',)       keys=[]
```

### 3.2 重放：从某一步重新执行

```python
# 找到"提纲已生成、正准备写文章"的那个存档
before_writing = next(s for s in history if s.next == ("write_article",))

# 传入 None + 该存档的 config，从这里重新执行（提纲不变，重新写文章）
result = graph.invoke(None, before_writing.config)
print(result["article"])
```

### 3.3 分叉：修改历史状态后执行

```python
# 在"写文章之前"的存档上，把提纲改掉
new_config = graph.update_state(
    before_writing.config,
    {"outline": "1. 远程办公节省通勤时间\n2. 但沟通成本上升\n3. 混合办公是折中方案"},
)

# 从修改后的新存档继续执行，按新提纲写文章
result = graph.invoke(None, new_config)
print(result["article"])
```

- `update_state` 不会修改原来的存档，而是**基于它创建一个新的 Checkpoint**，返回新的 config。
- 更新同样会经过 reducer：如果字段是 `add_messages` 类型，传入的消息会被追加（或按 ID 替换），而不是整体覆盖。

> 🎯 **应用场景**：
> - **调试**：Agent 在第 5 步做了错误决定，回到第 4 步看看当时的状态，改一下再试。
> - **产品功能**：聊天应用的"编辑之前的消息并重新生成"、"重新生成这条回答"。
> - **人工纠错**：AI 生成的中间结果（提纲、SQL、计划）有问题，人工修正后让它继续。

---

## 四、短期记忆管理：对话太长怎么办

多轮对话的消息会越积越多，带来三个问题：**费用增加、速度变慢、超出上下文窗口**。常见的三种处理办法：

| 策略 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| 裁剪（Trim） | 发给模型时只取最近 N 条 / N 个 Token | 简单、零额外成本 | 早期信息彻底丢失 |
| 删除（Delete） | 从状态中永久删除旧消息 | 状态体积小 | 同上 |
| 摘要（Summarize） | 把旧消息压缩成一段摘要 | 保留关键信息 | 需要额外的模型调用 |

### 4.1 裁剪：trim_messages

只影响**发给模型的内容**，状态中的完整历史保持不变：

```python
from langchain_core.messages.utils import count_tokens_approximately, trim_messages


def chatbot(state: MessagesState) -> dict:
    messages = trim_messages(
        state["messages"],
        strategy="last",                           # 保留最近的消息
        token_counter=count_tokens_approximately,  # 粗略估算 Token 数
        max_tokens=2000,                           # 最多保留 2000 Token
        start_on="human",                          # 裁剪后必须从用户消息开始
        end_on=("human", "tool"),                  # 以用户消息或工具消息结尾
    )
    return {"messages": [model.invoke(messages)]}
```

> ⚠️ **注意**：裁剪后的消息序列必须合法：多数模型要求以用户消息开头；带工具调用的 AI 消息后面必须紧跟对应的工具结果。`start_on` / `end_on` 参数就是用来保证这一点的。

### 4.2 删除：RemoveMessage

`add_messages` reducer 支持通过 `RemoveMessage` 删除指定 ID 的消息：

```python
from langchain.messages import RemoveMessage


def delete_old_messages(state: MessagesState) -> dict:
    messages = state["messages"]
    if len(messages) > 20:
        # 只保留最近 20 条
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:-20]]}
    return {}
```

清空全部消息：

```python
from langgraph.graph.message import REMOVE_ALL_MESSAGES

return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}
```

### 4.3 摘要：保留要点

```python
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

load_dotenv()

model = init_chat_model("deepseek:deepseek-flash", extra_body={"thinking": {"type": "disabled"}})
summarizer = init_chat_model("deepseek:deepseek-flash", extra_body={"thinking": {"type": "disabled"}})


class State(MessagesState):
    summary: str


def chatbot(state: State) -> dict:
    messages = state["messages"]
    if state.get("summary"):
        messages = [{"role": "system", "content": f"之前对话的摘要：{state['summary']}"}] + messages
    return {"messages": [model.invoke(messages)]}


def summarize(state: State) -> dict:
    existing = state.get("summary", "")
    prompt = (
        f"已有摘要：{existing}\n\n请结合以上摘要和下面的新对话，更新为一段不超过 200 字的摘要，保留用户的关键信息和诉求："
        if existing
        else "请把下面的对话总结为一段不超过 200 字的摘要，保留用户的关键信息和诉求："
    )
    new_summary = summarizer.invoke(state["messages"] + [{"role": "user", "content": prompt}]).content
    # 只保留最近 4 条消息，其余删除
    return {"summary": new_summary, "messages": [RemoveMessage(id=m.id) for m in state["messages"][:-4]]}


def need_summary(state: State) -> str:
    return "summarize" if len(state["messages"]) > 10 else END


builder = StateGraph(State)
builder.add_node(chatbot)
builder.add_node(summarize)
builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", need_summary, ["summarize", END])
builder.add_edge("summarize", END)
graph = builder.compile(checkpointer=InMemorySaver())
```

> 💡 如果你用的是 `create_agent`，直接使用第 07 章的 `SummarizationMiddleware` 即可，不用自己写。这里手写是为了理解原理，以及在自定义图中使用。

---

## 五、长期记忆：Store

### 5.1 为什么需要 Store

Checkpointer 的记忆**绑定在 thread 上**：用户开一个新会话，之前聊过的内容就都"忘了"。

但很多信息应该**跨会话**保留：

- 用户的偏好（"我喜欢简洁的回答"、"我对花生过敏"）
- 用户的基本信息（职业、所在城市）
- Agent 学到的经验（"这个用户的公司用的是 Vue 不是 React"）

> 📖 **术语**：**Store（存储）**——LangGraph 提供的跨 thread 的键值存储，用于实现长期记忆。数据按 **namespace（命名空间）** 组织。

> 📖 **术语**：**Namespace（命名空间）**——一个元组，用来给数据分组，类似文件夹路径。例如 `("users", "u001", "preferences")` 相当于 `/users/u001/preferences/`。

```mermaid
graph LR
    subgraph Threads[多个会话 Thread · Checkpointer]
        T1[周一的会话]
        T2[周三的会话]
        T3[周五的会话]
    end
    subgraph Store[Store · 长期记忆]
        N1["('memories', 'u001')<br/>喜欢简洁回答<br/>对花生过敏<br/>住在杭州"]
    end
    T1 -->|写入| N1
    N1 -->|读取| T2
    N1 -->|读取| T3
```

### 5.2 基本操作

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
namespace = ("memories", "u001")

# 写入：put(命名空间, 键, 值)，值必须是字典
store.put(namespace, "food", {"text": "对花生过敏，喜欢川菜"})
store.put(namespace, "style", {"text": "喜欢简洁、带要点的回答"})

# 按键读取
item = store.get(namespace, "food")
print(item.value)  # {'text': '对花生过敏，喜欢川菜'}

# 列出命名空间下的所有记忆
for item in store.search(namespace):
    print(item.key, item.value)

# 删除
store.delete(namespace, "style")
```

### 5.3 语义检索

记忆多了以后，每次都把全部记忆塞给模型不现实。给 Store 配置 Embedding 模型后，就可以**按语义检索最相关的记忆**：

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.store.memory import InMemoryStore

store = InMemoryStore(
    index={
        "embed": HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5"),  # 需要 uv add langchain-huggingface sentence-transformers
        "dims": 512,  # 向量维度，需与 Embedding 模型一致（bge-small-zh 为 512）
    }
)

ns = ("memories", "u001")
store.put(ns, "1", {"text": "对花生过敏"})
store.put(ns, "2", {"text": "是一名前端工程师，主要用 Vue"})
store.put(ns, "3", {"text": "周末喜欢爬山"})

results = store.search(ns, query="晚饭吃什么好？", limit=1)
print(results[0].value)  # {'text': '对花生过敏'}
```

> 💡 DeepSeek 不提供 Embedding 模型，这里和 [第 06 章](./06-RAG检索增强生成.md) 一样使用本地 BGE 模型。换成其他 Embedding 服务时，注意 `dims` 要和所选模型的维度一致。

生产环境使用持久化的 Store，如 `PostgresStore`（`from langgraph.store.postgres import PostgresStore`），用法与 `InMemoryStore` 一致。

### 5.4 在图中使用 Store

编译时传入 `store`，节点通过 `runtime.store` 访问；用户 ID 等信息通过 context 传入：

```python
# long_term_memory_bot.py
import uuid
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

load_dotenv()

model = init_chat_model("deepseek:deepseek-flash", extra_body={"thinking": {"type": "disabled"}})
extractor = init_chat_model("deepseek:deepseek-flash", extra_body={"thinking": {"type": "disabled"}}, temperature=0)


@dataclass
class Context:
    user_id: str


class NewMemories(BaseModel):
    facts: list[str] = Field(
        default_factory=list,
        description="从用户最新消息中提取的、值得长期记住的个人信息或偏好；没有则为空列表",
    )


def chatbot(state: MessagesState, runtime: Runtime[Context]) -> dict:
    namespace = ("memories", runtime.context.user_id)
    memories = runtime.store.search(namespace)
    memory_text = "\n".join(f"- {m.value['text']}" for m in memories) or "（暂无）"
    system = f"你是贴心的私人助理。以下是你记住的关于用户的信息：\n{memory_text}"
    reply = model.invoke([{"role": "system", "content": system}] + state["messages"])
    return {"messages": [reply]}


def save_memories(state: MessagesState, runtime: Runtime[Context]) -> dict:
    last_user_msg = next(m for m in reversed(state["messages"]) if m.type == "human")
    result = extractor.with_structured_output(NewMemories).invoke(
        f"用户说：{last_user_msg.content}\n提取值得长期记住的用户信息（如姓名、职业、偏好、禁忌）。"
    )
    namespace = ("memories", runtime.context.user_id)
    for fact in result.facts:
        runtime.store.put(namespace, str(uuid.uuid4()), {"text": fact})
    return {}


builder = StateGraph(MessagesState, context_schema=Context)
builder.add_node(chatbot)
builder.add_node(save_memories)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", "save_memories")
builder.add_edge("save_memories", END)

graph = builder.compile(checkpointer=InMemorySaver(), store=InMemoryStore())
ctx = Context(user_id="u001")

# 会话 1
graph.invoke(
    {"messages": [{"role": "user", "content": "我叫小林，对海鲜过敏，最近在减脂"}]},
    {"configurable": {"thread_id": "monday"}},
    context=ctx,
)

# 会话 2：全新的 thread，但依然记得用户信息
result = graph.invoke(
    {"messages": [{"role": "user", "content": "今晚推荐我吃点什么？"}]},
    {"configurable": {"thread_id": "friday"}},
    context=ctx,
)
print(result["messages"][-1].content)  # 会避开海鲜，并推荐低脂餐
```

> ⚠️ **注意**：
> - 长期记忆一定要按**用户隔离**（namespace 里带上 user_id），否则 A 用户的隐私会泄露给 B 用户。
> - 写入记忆前最好做去重和冲突处理（例如用户之前说"住北京"，现在说"搬到上海了"），生产中常用"让模型对比已有记忆后决定新增 / 更新 / 删除"的方式。
> - `create_agent` 也支持 `store` 参数，工具里通过 `runtime.store` 访问（见第 07 章 `ToolRuntime`）。

> 🎯 **应用场景**：私人助理、AI 陪伴、个性化推荐、教育辅导（记住学生的薄弱知识点）、企业 Agent 沉淀团队知识（"这个客户合同里约定了 30 天账期"）。

---

## 六、人机协同：interrupt

### 6.1 为什么需要人机协同

完全自动化的 AI 在这些场景下是危险的：

- **高风险操作**：付款、删除数据、发送对外邮件；
- **结果需要把关**：AI 生成的合同条款、对外公告；
- **信息不足**：用户说"帮我订机票"，但没说日期。

> 📖 **术语**：**interrupt（中断）**——在节点中调用 `interrupt(...)`，图会**暂停执行并保存状态**，把信息返回给调用方；等人处理完后，调用方用 **`Command(resume=...)`** 恢复，`interrupt(...)` 的返回值就是人给出的答复。

```mermaid
sequenceDiagram
    participant App as 你的程序
    participant G as LangGraph 图
    participant H as 人
    App->>G: invoke(输入, thread_id=1)
    G->>G: 执行节点 A、B
    G->>G: 节点 C 调用 interrupt("是否批准?")
    G-->>App: 返回 result["__interrupt__"]（状态已存档）
    App->>H: 展示审批界面
    H-->>App: 批准
    App->>G: invoke(Command(resume="批准"), thread_id=1)
    G->>G: 节点 C 从头重新执行，interrupt 返回"批准"
    G->>G: 继续执行后续节点
    G-->>App: 最终结果
```

使用前提：

1. 编译时必须有 **checkpointer**（暂停时要存档）；
2. 恢复时必须使用**同一个 `thread_id`**。

### 6.2 场景一：审批

```python
from typing import Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class RefundState(TypedDict):
    order_id: str
    amount: float
    approved: bool
    result: str


def check_refund(state: RefundState) -> dict:
    # 实际项目里这里可能是 AI 分析退款理由、校验订单
    return {}


def human_approval(state: RefundState) -> dict:
    decision = interrupt(
        {
            "question": "是否批准此退款？",
            "order_id": state["order_id"],
            "amount": state["amount"],
        }
    )
    return {"approved": decision == "approve"}


def route(state: RefundState) -> Literal["do_refund", "reject_refund"]:
    return "do_refund" if state["approved"] else "reject_refund"


def do_refund(state: RefundState) -> dict:
    return {"result": f"✅ 订单 {state['order_id']} 已退款 ¥{state['amount']}"}


def reject_refund(state: RefundState) -> dict:
    return {"result": f"❌ 订单 {state['order_id']} 退款被拒绝"}


def needs_approval(state: RefundState) -> Literal["human_approval", "do_refund"]:
    # 小额自动通过，大额才需要人工审批
    return "human_approval" if state["amount"] > 500 else "do_refund"


builder = StateGraph(RefundState)
builder.add_node(check_refund)
builder.add_node(human_approval)
builder.add_node(do_refund)
builder.add_node(reject_refund)
builder.add_edge(START, "check_refund")
builder.add_conditional_edges("check_refund", needs_approval)
builder.add_conditional_edges("human_approval", route)
builder.add_edge("do_refund", END)
builder.add_edge("reject_refund", END)
graph = builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "refund-A1002"}}
result = graph.invoke({"order_id": "A1002", "amount": 1899.0}, config)

if "__interrupt__" in result:
    payload = result["__interrupt__"][0].value
    print(f"⏸️ {payload['question']} 订单 {payload['order_id']}，金额 ¥{payload['amount']}")
    decision = input("输入 approve / reject：")
    result = graph.invoke(Command(resume=decision), config)

print(result["result"])
```

> 🎯 **应用场景**：财务审批、内容发布审核、生产环境变更审批、合同签署前确认。在 Web 应用里，`__interrupt__` 的内容会推送给前端渲染成审批卡片，审批人点击按钮后由后端调用 `Command(resume=...)`。暂停期间状态存在数据库里，**可以等几分钟，也可以等几天**。

### 6.3 场景二：审核并编辑 AI 的产出

```python
def review_draft(state: State) -> dict:
    edited = interrupt(
        {
            "instruction": "请审核 AI 生成的邮件，可直接修改后提交",
            "draft": state["draft"],
        }
    )
    return {"draft": edited}  # 用人修改后的版本覆盖


# 调用方：
# result = graph.invoke(inputs, config)
# draft = result["__interrupt__"][0].value["draft"]
# ……人在界面上修改……
# graph.invoke(Command(resume=修改后的文本), config)
```

> 💡 另一种做法是暂停后用第三节的 `graph.update_state(config, {...})` 修改状态，再 `graph.invoke(None, config)` 继续。`interrupt` 的方式把"在哪里需要人工介入"写在了流程里，更清晰。

### 6.4 场景三：信息不全时追问用户

用户说"帮我订一张机票"，缺少出发地、目的地、日期。让 AI 检查缺什么、追问，直到信息完整：

```mermaid
graph TD
    START([用户需求]) --> E[extract<br/>提取已有信息]
    E --> C{信息齐全?}
    C -->|否| ASK[ask_user<br/>interrupt 追问]
    ASK --> E
    C -->|是| BOOK[book<br/>执行预订]
    BOOK --> END([结束])
```

```python
from typing import Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel

load_dotenv()

model = init_chat_model("deepseek:deepseek-flash", extra_body={"thinking": {"type": "disabled"}}, temperature=0)


class FlightInfo(BaseModel):
    from_city: Optional[str] = None
    to_city: Optional[str] = None
    date: Optional[str] = None


class State(TypedDict):
    conversation: str  # 累积的用户输入
    info: dict
    result: str


def extract(state: State) -> dict:
    info = model.with_structured_output(FlightInfo).invoke(
        f"从以下对话中提取出发城市、到达城市、日期，未提及的留空：\n{state['conversation']}"
    )
    return {"info": info.model_dump()}


def is_complete(state: State) -> Literal["ask_user", "book"]:
    return "book" if all(state["info"].values()) else "ask_user"


def ask_user(state: State) -> dict:
    labels = {"from_city": "出发城市", "to_city": "到达城市", "date": "出发日期"}
    missing = [labels[k] for k, v in state["info"].items() if not v]
    answer = interrupt(f"请补充：{'、'.join(missing)}")
    return {"conversation": state["conversation"] + "\n" + answer}


def book(state: State) -> dict:
    i = state["info"]
    return {"result": f"✈️ 已为您预订 {i['date']} {i['from_city']} → {i['to_city']} 的航班"}


builder = StateGraph(State)
builder.add_node(extract)
builder.add_node(ask_user)
builder.add_node(book)
builder.add_edge(START, "extract")
builder.add_conditional_edges("extract", is_complete)
builder.add_edge("ask_user", "extract")  # 用边形成追问循环，而不是在节点里写 while
builder.add_edge("book", END)
graph = builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "flight-1"}}
result = graph.invoke({"conversation": "帮我订一张去成都的机票"}, config)

while "__interrupt__" in result:
    reply = input(f"🤖 {result['__interrupt__'][0].value}\n👤 ")
    result = graph.invoke(Command(resume=reply), config)

print(result["result"])
```

### 6.5 interrupt 的五个关键注意事项

> ⚠️ **注意（务必理解，否则会踩坑）**：
>
> **① 恢复时，节点会从头重新执行。** 不是从 `interrupt()` 那一行继续，而是整个节点函数重新跑一遍，执行到 `interrupt()` 时直接返回 resume 的值。所以 **`interrupt()` 之前的代码会执行两次**。
>
> **② 副作用要幂等，或放到 interrupt 之后。** 如果 `interrupt()` 之前有"扣款""发邮件""插入数据库"等操作，恢复时会再做一遍。解决办法：把副作用放到 `interrupt()` 之后，或拆到单独的节点里，或使用 upsert 等幂等操作。
>
> **③ 不要用宽泛的 try/except 包住 interrupt。** `interrupt` 是通过抛出一个特殊异常来实现暂停的，`except Exception` 会把它吞掉，导致暂停失效。
>
> **④ 一个节点里多次 interrupt 时，顺序必须固定。** 恢复值是按调用顺序（索引）匹配的，不要根据条件跳过某些 `interrupt()`。需要反复询问时，用条件边形成循环（如 6.4），不要在节点里写 `while True`。
>
> **⑤ interrupt 的参数要可 JSON 序列化**（字符串、数字、字典、列表），不要传函数或类实例。

用代码直观感受第①点：

```python
def node_with_side_effect(state):
    print("📧 发送通知邮件")       # ❌ 恢复时会再打印一次（再发一次邮件）
    answer = interrupt("确认吗？")
    print("💳 执行扣款")           # ✅ 只在恢复后执行一次
    return {"answer": answer}
```

---

## 七、Command：在节点内同时"更新状态 + 跳转"

到目前为止，"更新状态"靠节点返回值，"决定去哪"靠条件边，两者是分开的。`Command` 可以让节点**一步完成两件事**：

```python
from typing import Literal

from langgraph.types import Command


def risk_check(state: State) -> Command[Literal["auto_approve", "manual_review"]]:
    score = calculate_risk(state)
    if score > 80:
        return Command(update={"risk_score": score, "reason": "高风险"}, goto="manual_review")
    return Command(update={"risk_score": score}, goto="auto_approve")
```

- 返回类型注解 `Command[Literal[...]]` **必须写**，LangGraph 靠它知道可能的去向（用于校验和画图）。
- 使用 `Command` 跳转的节点，**不需要再写条件边**。
- `Command` 只是增加动态跳转，如果同一个节点还有 `add_edge` 定义的静态边，静态边仍然会执行。一个节点只用一种方式。

| 方式 | 适合 |
|---|---|
| 条件边 + 路由函数 | 路由逻辑独立、简单，只依赖状态 |
| `Command` | 跳转决策和状态更新紧密相关（比如"判断结果"和"判断原因"都要保存），以及多 Agent 之间的"交接"（handoff，第 10 章） |

> ⚠️ **注意**：作为 `invoke` 的输入时，`Command` 只用于 `Command(resume=...)` 恢复中断。不要用 `Command(update=...)` 作为输入来"继续对话"，继续对话直接传新消息即可。

---

## 八、综合示例：带审批与长期记忆的周报助手

把本章内容整合起来：

- 根据用户的工作记录生成周报（AI）
- 主管风格偏好存在长期记忆里（Store）
- 生成后交给用户审核编辑（interrupt）
- 用户的修改意见会被沉淀为新的风格偏好（Store）
- 整个过程可中断、可恢复（Checkpointer）

```mermaid
graph TD
    START([本周工作记录]) --> G[generate<br/>读取风格偏好, 生成周报]
    G --> R[review<br/>interrupt: 用户审核]
    R -->|通过| S[save<br/>保存定稿]
    R -->|提出修改意见| L[learn<br/>修改意见存入长期记忆]
    L --> G
    S --> END([结束])
```

```python
# weekly_report_assistant.py
import uuid
from dataclasses import dataclass
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, interrupt

load_dotenv()

model = init_chat_model("deepseek:deepseek-flash", extra_body={"thinking": {"type": "disabled"}})


@dataclass
class Context:
    user_id: str


class State(TypedDict):
    work_log: str
    report: str
    feedback: str


def generate(state: State, runtime: Runtime[Context]) -> dict:
    prefs = runtime.store.search(("report_style", runtime.context.user_id))
    style = "\n".join(f"- {p.value['rule']}" for p in prefs) or "- 结构清晰即可"
    prompt = f"""根据工作记录写一份周报。
必须遵守的写作偏好：
{style}

工作记录：
{state['work_log']}"""
    if state.get("feedback"):
        prompt += f"\n\n上一版：\n{state['report']}\n\n修改意见：{state['feedback']}"
    return {"report": model.invoke(prompt).content}


def review(state: State) -> Command[Literal["save", "learn"]]:
    answer = interrupt({"report": state["report"], "hint": "输入 ok 通过，或直接输入修改意见"})
    if answer.strip().lower() == "ok":
        return Command(goto="save")
    return Command(update={"feedback": answer}, goto="learn")


def learn(state: State, runtime: Runtime[Context]) -> dict:
    # 把修改意见沉淀为长期偏好，下周写周报时自动遵守
    rule = model.invoke(
        f"把这条周报修改意见概括为一条通用的写作规则（20 字以内）：{state['feedback']}"
    ).content
    runtime.store.put(("report_style", runtime.context.user_id), str(uuid.uuid4()), {"rule": rule})
    print(f"🧠 已记住偏好：{rule}")
    return {}


def save(state: State) -> dict:
    print("💾 周报已保存")
    return {}


builder = StateGraph(State, context_schema=Context)
builder.add_node(generate)
builder.add_node(review)
builder.add_node(learn)
builder.add_node(save)
builder.add_edge(START, "generate")
builder.add_edge("generate", "review")
builder.add_edge("learn", "generate")
builder.add_edge("save", END)

store = InMemoryStore()
graph = builder.compile(checkpointer=InMemorySaver(), store=store)


def run_week(thread_id: str, work_log: str):
    config = {"configurable": {"thread_id": thread_id}}
    ctx = Context(user_id="u001")
    result = graph.invoke({"work_log": work_log}, config, context=ctx)
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print(f"\n------ 周报草稿 ------\n{payload['report']}\n----------------------")
        answer = input(f"{payload['hint']}：")
        result = graph.invoke(Command(resume=answer), config, context=ctx)
    return result["report"]


if __name__ == "__main__":
    run_week("week-41", "完成登录页重构；修复 3 个线上 bug；参加技术分享")
    # 第一周可以提意见，比如“太啰嗦了，用要点列出，并加上下周计划”
    run_week("week-42", "接入支付 SDK；优化首屏加载 40%；面试 2 位候选人")
    # 第二周的初稿会自动遵守第一周学到的偏好
```

---

## 本章小结

- **Checkpointer** 在每一步后自动存档，按 `thread_id` 区分会话，带来四种能力：多轮对话、断点续跑、时间旅行、人机协同。开发用 `InMemorySaver`，生产用 `PostgresSaver` 等持久化实现。
- `get_state` 查看当前状态，`get_state_history` 查看历史；`invoke(None, 某个旧config)` 重放，`update_state` 修改后分叉。
- 短期记忆过长时：`trim_messages` 裁剪、`RemoveMessage` 删除、摘要压缩。
- **Store** 提供跨会话的长期记忆，按 namespace 组织，支持语义检索；节点通过 `runtime.store` 访问；务必按用户隔离。
- **`interrupt()` + `Command(resume=...)`** 实现人机协同：审批、编辑、追问。牢记：恢复时节点从头重跑、副作用要幂等、不要用宽泛 try/except、同一 thread_id。
- **`Command(update=..., goto=...)`** 让节点同时更新状态和决定下一步。

## 练习

1. 把 2.2 节的聊天机器人改用 `SqliteSaver`，聊几句后关闭程序，重新运行并用同一个 `thread_id` 继续聊，验证记忆仍在。
2. 在 3.1 节的示例中，用 `update_state` 把 `topic` 改成另一个主题，从最开始的存档重新执行，对比两次生成的文章。
3. 给 6.2 节的退款流程增加"审批人可以修改退款金额"的能力（提示：resume 时传入字典 `{"decision": "approve", "amount": 1500}`）。
4. 观察 interrupt 的"重跑"特性：在 `human_approval` 节点的 `interrupt()` 前后各加一行 `print`，运行并恢复，数一数各打印了几次。

<details>
<summary>参考思路</summary>

1. `SqliteSaver(sqlite3.connect("chat.db", check_same_thread=False))`，其余代码不变。
2. 在历史中找到 `next == ("make_outline",)` 的存档，`update_state(snap.config, {"topic": "新主题"})` 得到新 config，`invoke(None, new_config)`。
3. `decision = interrupt(...)` 得到字典，返回 `{"approved": decision["decision"] == "approve", "amount": decision.get("amount", state["amount"])}`。
4. `interrupt()` 之前的 print 执行 2 次（首次运行 + 恢复时重跑），之后的 print 执行 1 次。

</details>

---

**上一章** ← [08 LangGraph 核心概念](./08-LangGraph核心概念.md) ｜ **下一章** → [10 高级模式：流式、子图与多 Agent](./10-LangGraph高级模式-流式子图多Agent.md)
