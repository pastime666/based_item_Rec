import os
import random
import numpy as np
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import deepseek_api
# ========== 配置部分 ==========
# 本地 Qwen3-0.6B 模型目录
LOCAL_MODEL_PATH = "/root/models/PretrainedModel/qwen3-0.6b"  
LOCAL_MODEL_PATH = os.path.expanduser(LOCAL_MODEL_PATH)
assert os.path.isdir(LOCAL_MODEL_PATH), f"本地模型目录不存在: {LOCAL_MODEL_PATH}"
print("本地模型目录文件列表：", os.listdir(LOCAL_MODEL_PATH))

# 超参：根据数据规模和显存自行调整
M = 100      # item 数量（模拟）
U = 300     # user 数量（模拟）
MARKOV_ITERS = 5
AMPLIFY_POWER = 2.0
PRUNE_THRESHOLD = 1e-3
TOP_K = 5     # Markov 提取强关联时每行取 top_k
MAX_PRED_USERS = 20
BATCH_SIZE = 8
NUM_NEG = 1   # BPR 每个正对采多少负样本
LR = 1e-4
NUM_EPOCHS = 10
MAX_SEQ_LEN = 256

# ========== 1. 准备 item–user 交互数据 ==========
clicks = defaultdict(list)  # item_id -> list of user_id
for u in range(U):
    for i in random.sample(range(M), random.randint(5, 30)):
        clicks[i].append(u)
item_users = {i: set(us) for i, us in clicks.items()}  # item -> set(users)

# ========== 2. 构造初始 item–item 转移概率矩阵 P0 ==========
P0 = np.zeros((M, M), dtype=np.float32)
for i in range(M):
    ui = item_users.get(i, set())
    for j in range(M):
        if i == j:
            continue
        uj = item_users.get(j, set())
        inter = len(ui & uj)
        P0[i, j] = float(inter)
    s = P0[i].sum()
    if s > 0:
        P0[i] /= s

# ========== 3. Markov 迭代 + 扩张/剪枝，得到稳态 P_steady ==========
def iterate_markov(P_init, num_iters=MARKOV_ITERS, amplify_power=AMPLIFY_POWER, prune_threshold=PRUNE_THRESHOLD):
    P_t = P_init.copy()
    for _ in range(num_iters):
        P_next = P_t @ P_t
        # 行归一化
        row_sums = P_next.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        P_next = P_next / row_sums
        # 放大/压缩
        P_next = np.power(P_next, amplify_power)
        # 阈值剪枝
        P_next[P_next < prune_threshold] = 0.0
        # 再归一化
        row_sums = P_next.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        P_t = P_next / row_sums
    return P_t

P_steady = iterate_markov(P0)

# ========== 4. 提取强关联 item 对，预测用户 ==========
predicted_users = {}
for i in range(M):
    row = P_steady[i].copy()
    row[i] = 0.0
    nbrs = row.argsort()[-TOP_K:]
    cand = set()
    for j in nbrs:
        cand |= item_users.get(j, set())
    # 限制预测用户数量
    if len(cand) > MAX_PRED_USERS:
        cand = set(random.sample(list(cand), MAX_PRED_USERS))
    predicted_users[i] = cand

# ========== 5. Prompt 设计 ==========
def get_item_base_info(item_id,f):
    
    
    title = f"Item {item_id} Title"
    abstract = f"This is a brief abstract of item {item_id}, describing its content."
    category = f"Category_{item_id % 10}"
    return title, abstract, category

def call_closed_llm_to_enrich(description):
    return deepseek_api.generate_item_metadata_with_deepseek(description)
    

item_prompts = []
f=open('item.txt','r')
descriptions=[]
lines=f.readlines()
for line in lines:
    line=line.strip()
    if line:
        descriptions.append(line)
f.close()
for i in range(M):
    description = descriptions[i]
    enriched_info = call_closed_llm_to_enrich(description)
    u_exp = sorted(item_users.get(i, []))
    u_pred = sorted(predicted_users.get(i, []))
    users_field = ",".join(map(str, u_exp + u_pred))
    prompt = f"{enriched_info}. Users: {users_field}"
    item_prompts.append(prompt)

# ========== 6. 加载 Qwen3-0.6B，本地加载并冻结/LoRA ==========
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)

# 使用 BitsAndBytesConfig 代替 load_in_8bit
quant_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH,
    trust_remote_code=True,
    device_map="auto",
    quantization_config=quant_config
)

# 冻结所有参数
for param in model.parameters():
    param.requires_grad = False


# 共 28 层，索引 0~27，取最后两层 26,27
last_layer_indices = [26, 27]
for name, param in model.named_parameters():
    # 仅对浮点类型参数解冻
    if any(f"layers.{i}" in name for i in last_layer_indices) and param.dtype.is_floating_point:
        param.requires_grad = True

# 插入 LoRA，仅在最后两层的 q_proj/v_proj 上
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
# 仅 LoRA 参数是可训练的

# ========== 7. 定义 BPRPairDataset 与 collate_fn ==========
class BPRPairDataset(Dataset):
    def __init__(self, item_prompts, item_users, num_items, num_neg=NUM_NEG):
        self.item_prompts = item_prompts
        self.item_users = item_users
        # user -> items
        self.user_items = {}
        for i, users in item_users.items():
            for u in users:
                self.user_items.setdefault(u, []).append(i)
        # 仅保留历史 >=2 的用户
        self.users = [u for u, his in self.user_items.items() if len(his) >= 2]
        self.num_items = num_items
        self.num_neg = num_neg

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        his = self.user_items[u]
        i, j_pos = random.sample(his, 2)
        negs = []
        while len(negs) < self.num_neg:
            j = random.randrange(self.num_items)
            if j not in his:
                negs.append(j)
        return i, j_pos, negs

def collate_bpr(batch, tokenizer, prompts, device, max_length=MAX_SEQ_LEN):
    anchors, positives, negatives = [], [], []
    for i, j_pos, negs in batch:
        anchors.append(prompts[i])
        positives.append(prompts[j_pos])
        for neg in negs:
            negatives.append(prompts[neg])
    all_texts = anchors + positives + negatives
    enc = tokenizer(all_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    B = len(batch)
    num_neg = len(batch[0][2])
    anchor_ids = enc["input_ids"][:B].to(device)
    anchor_mask = enc["attention_mask"][:B].to(device)
    pos_ids = enc["input_ids"][B: B*2].to(device)
    pos_mask = enc["attention_mask"][B: B*2].to(device)
    neg_ids = enc["input_ids"][B*2:].to(device)  # [B*num_neg, L]
    neg_mask = enc["attention_mask"][B*2:].to(device)
    # reshape negatives 为 [B, num_neg, L]
    neg_ids = neg_ids.view(B, num_neg, -1)
    neg_mask = neg_mask.view(B, num_neg, -1)
    return anchor_ids, anchor_mask, pos_ids, pos_mask, neg_ids, neg_mask

# ========== 8. get_embedding 与 BPR loss ==========
def get_embedding(input_ids, attention_mask):
    outputs = model.base_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True
    )
    last_h = outputs.hidden_states[-1]  # [B, L, H]
    mask = attention_mask.unsqueeze(-1)  # [B, L, 1]
    summed = (last_h * mask).sum(dim=1)  # [B, H]
    lengths = mask.sum(dim=1)  # [B,1]
    lengths = lengths.clamp(min=1)
    emb = summed / lengths  # [B, H]
    return emb

def bpr_loss(anchor_emb, pos_emb, neg_embs):
    anchor_norm = anchor_emb / (anchor_emb.norm(dim=1, keepdim=True) + 1e-12)
    pos_norm = pos_emb / (pos_emb.norm(dim=1, keepdim=True) + 1e-12)
    neg_norm = neg_embs / (neg_embs.norm(dim=2, keepdim=True) + 1e-12)  # [B,N,H]
    sim_pos = (anchor_norm * pos_norm).sum(dim=1)  # [B]
    sim_negs = torch.einsum("bh,bnh->bn", anchor_norm, neg_norm)  # [B, N]
    diff = sim_pos.unsqueeze(1) - sim_negs  # [B, N]
    loss = -torch.log(torch.sigmoid(diff) + 1e-12).mean()
    return loss

# ========== 9. 训练 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

dataset = BPRPairDataset(item_prompts, item_users, num_items=M, num_neg=NUM_NEG)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                    collate_fn=lambda b: collate_bpr(b, tokenizer, item_prompts, device))

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    for anchor_ids, anchor_mask, pos_ids, pos_mask, neg_ids, neg_mask in loader:
        anchor_emb = get_embedding(anchor_ids, anchor_mask)  # [B,H]
        pos_emb = get_embedding(pos_ids, pos_mask)          # [B,H]
        B_cur, N, L = neg_ids.shape
        neg_ids_flat = neg_ids.view(B_cur*N, L)
        neg_mask_flat = neg_mask.view(B_cur*N, L)
        neg_emb_flat = get_embedding(neg_ids_flat, neg_mask_flat)  # [B*N, H]
        neg_embs = neg_emb_flat.view(B_cur, N, -1)                # [B, N, H]

        loss = bpr_loss(anchor_emb, pos_emb, neg_embs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg = total_loss / len(loader) if len(loader)>0 else 0.0
    print(f"Epoch {epoch}: avg BPR loss = {avg:.4f}")

# ========== 10. 训练后生成 item embedding 并评估 Recall@N ==========
model.eval()
all_embs = []
batch_size = 16
with torch.no_grad():
    for i in range(0, M, batch_size):
        batch_prompts = item_prompts[i: i+batch_size]
        enc = tokenizer(batch_prompts, padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt").to(device)
        emb = get_embedding(enc["input_ids"], enc["attention_mask"])  # [bs, H]
        all_embs.append(emb.cpu())
all_embs = torch.cat(all_embs, dim=0)  # [M, H]
emb_norm = all_embs / (all_embs.norm(dim=1, keepdim=True) + 1e-12)

def eval_recall(all_embs_norm, item_users, N=5):
    recalls = []
    M0 = all_embs_norm.size(0)
    for u in range(U):
        his = [i for i in range(M0) if u in item_users.get(i, [])]
        if len(his) < 2:
            continue
        q = random.choice(his)
        gt = set(his) - {q}
        if not gt:
            continue
        sim = all_embs_norm[q].unsqueeze(0) @ all_embs_norm.T  # [1, M]
        sim = sim.squeeze(0).numpy()
        sim[q] = -1.0
        topn = np.argsort(sim)[-N:][::-1]
        hit = len(set(topn) & gt)
        recalls.append(hit / min(N, len(gt)))
    return float(np.mean(recalls)) if recalls else 0.0

for N in [1, 5, 10]:
    r = eval_recall(emb_norm, item_users, N)
    print(f"Recall@{N} = {r:.4f}")

