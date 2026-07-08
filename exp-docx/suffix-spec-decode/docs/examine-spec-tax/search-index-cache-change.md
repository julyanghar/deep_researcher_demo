# 修改记录:search 块索引加进程级缓存(带内存 LRU 上限)

- 日期:2026-07-07
- 文件:[search.py](../../../../deep_researcher_demo/search.py)(仅此一个)
- 动机来源:[e2e-search-bottleneck.md](e2e-search-bottleneck.md)(为什么搜索占 e2e 32%、怎么排除限流、怎么定位真因)
- 验收物证:[review-report.md](../../../../../modify-code-runs/search-index-cache/review-report.md)(5 条验收 + 证据地址)

## 一句话

同一道题的"块索引"(`chunks.jsonl`,23MB)本来**每次 `search()` 都被从头读+解析,一道题 ~36 遍**;
改成**解析一次就缓存复用**,并给缓存加了 **16GB 内存 LRU 上限**。检索结果一字未变,搜索窗口 34s→~10s。

> "块索引"= 每道题联网抓的 ~100 个网页,切成几千小块、每块配一个 1024 维 embedding 向量,连同文字、来源网址
> 一起存进 `chunks.jsonl`。检索就是拿查询向量在这个文件里挑最相近的几块。q3 这题有 3455 块。

## 二、改了什么(4 处)

### 1. 加模块级缓存 + 内存上限([search.py:315-333](../../../../deep_researcher_demo/search.py#L315-L333))
```python
_CHUNK_INDEX_CACHE: "OrderedDict[str, tuple]" = OrderedDict()   # path -> (sig, records, nbytes)
_CHUNK_INDEX_LOCK = threading.Lock()
_CHUNK_CACHE_MAX_BYTES = int(float(os.getenv("CHUNK_CACHE_MAX_GB", "16")) * (1024 ** 3))
def _estimate_records_bytes(records): ...   # 估一份索引常驻内存的字节(大头是每块 1024 个浮点)
```
- `OrderedDict` 是为了做 LRU(记住谁最久没用)。
- **独立锁** `_CHUNK_INDEX_LOCK`,不复用已有的 `_CACHE_LOCK`:因为 `_append_chunk_index` 持 `_CACHE_LOCK` 时会调到
  `_load_chunk_index`,共用一把锁会自己锁死自己(重入)。锁序恒为 `_CACHE_LOCK → _CHUNK_INDEX_LOCK`,不会有逆序死锁。

### 2. `_load_chunk_index` 改成"先查缓存"([search.py:550-580](../../../../deep_researcher_demo/search.py#L550-L580))
- 用 **(文件路径, 修改时间 `st_mtime_ns`, 文件大小 `st_size`)** 当缓存钥匙(签名)。
- **命中**(签名没变):`move_to_end` 标记"刚用过",直接返回缓存里的对象,**不碰文件**。
- **未命中**(第一次 / 文件变了):真读文件解析一遍(这里打 `MCDBG_parse` 日志),存进缓存。
- 存完做 **LRU 驱逐**:所有缓存项的内存加起来超 `_CHUNK_CACHE_MAX_BYTES` 就 `popitem(last=False)` 踢掉最久没用的,
  但 `len > 1` 才踢——**至少保留刚加载的当前题**,不然刚存就被踢掉、白干。

### 3. `_indexed_urls` 改成从缓存派生([search.py:546-548](../../../../deep_researcher_demo/search.py#L546-L548))
```python
return {r["url"] for r in self._load_chunk_index() if "url" in r}
```
原本它自己独立把 23MB 又全解析一遍(只为拿网址集合)。改成复用 `_load_chunk_index` 的缓存 → 顺手把
`_ensure_chunk_index` 里那 2.5s 也省了。

### 4. import 补两个([search.py:3-13](../../../../deep_researcher_demo/search.py#L3-L13)):`sys`(估内存)、`from collections import OrderedDict`(LRU)。

## 三、为什么这么设计(几个关键取舍)

- **为什么按 (mtime, size) 当钥匙**:replay 跑实验时索引根本不变 → 签名不变 → 每次命中缓存,零解析。而 online 录制时
  `_append_chunk_index` 会往文件追加内容 → 文件 mtime/size 变 → 签名对不上 → 自动重解析,**读到的永远是最新文件**,
  正确性不靠人工失效、靠签名兜底。
- **为什么用内存上限而不是"缓存几道题"**:题的大小差很多(q18 有 940 块、q3 有 3455 块),按题数卡不准内存;
  按字节卡才是真正盯住"别把内存吃爆"。默认 **16GB**:实测每题索引 ~118MB,16GB 约容 ~138 题。
  而 run 是**逐题串行**、任一时刻只需 1~2 题常驻 → **16GB 正常永远碰不到,是防失控 OOM 的安全天花板**。
  想更保守设 `CHUNK_CACHE_MAX_GB=2` 也够用。
- **为什么"至少保留当前题"**:驱逐循环 `while len>1 and 超上限`。假如某题单独就超上限,也不会把它自己踢掉
  (否则这次 search 拿不到索引),只是不再多留别的题。
- **没做 numpy 余弦**:算相似度那步(纯 Python `sum(x*y)`)其实也慢,numpy 能快很多,但它靠
  `sorted((score, idx), reverse=True)` 的精确排序定 top-k,numpy 的浮点微小误差会**翻转两个分数接近的块的先后 →
  改动检索结果**。只省 ~11% 不值这个险,留作"日后要做需逐位校验"。

## 四、效果(改前 → 改后)

| 指标 | 改前 | 改后 |
|---|---|---|
| 一道题内索引解析次数 | ~36 遍 | **1 遍** |
| 单次 `search()`(3 子查询) | 12.2s | **~7.5s** |
| 3 个 researcher 并发窗口 | 35.1s | **~10s(3.5×)** |
| 检索 top-k 结果 | — | **逐字节一致(没变)** |
| 每题索引常驻内存 | — | ~118MB(16GB 上限约容 138 题) |

## 五、怎么用/调(环境变量)

| env | 默认 | 作用 |
|---|---|---|
| `CHUNK_CACHE_MAX_GB` | `16` | 缓存内存上限(GB);超了踢最久没用的题 |
| `MCDBG` | 关 | 设为任意非空值 → 每次真解析索引打一行 `MCDBG_parse <路径>`,用来核对缓存有没有生效 |

## 六、行为不变的保证

命门是"提速不能改检索结果"。验收 A1:改后 `_load_chunk_index`/`_indexed_urls` 与"独立从头全解析"逐值比对 =
`IDENTICAL`(3455=3455)。因为这次只动了"索引怎么加载"这一层,**算相似度、取 top-k 的逻辑一行没碰** → 端到端检索必然一致。
详见 [review-report.md 证据索引表](../../../../../modify-code-runs/search-index-cache/review-report.md)。

## 相关

- 真因调查:[e2e-search-bottleneck.md](e2e-search-bottleneck.md)
- decode 侧算力拆解:[decode-step-compute-anatomy.md](decode-step-compute-anatomy.md)
