import os
import pandas as pd
import tiktoken
import time
import asyncio
from datetime import datetime, timedelta
from openai import AsyncOpenAI

PROJECTS = [
    "spark",
    "flink",
    "hbase",
    "hive",
    "camel",
    "ignite",
    "cassandra",
    "hadoop",
    "solr",
    "hdfs",
    "kafka",
    "impala",
    "oak",
    "beam",
    "geode",
    "yarn",
    "lucene",
    "karaf",
    "wicket",
    "derby",
    "drill",
    "phoenix",
    "pdfbox",
    "mapreduce"
]

MODEL_NAME = 'gpt-4o-mini'
MAX_INPUT_TOKENS = 3000
MAX_RETRIES = 3
REQUEST_DELAY = 0.1
CONCURRENCY = 20
BATCH_SIZE = 5
LOG_FREQUENCY = 20
BASE_DATA_PATH = "code/LHGLink/data"

# 初始化tokenizer
tokenizer = tiktoken.encoding_for_model("gpt-4")

# 异步OpenAI客户端
def get_async_openai_client():
    return AsyncOpenAI(
        base_url=" ",
        api_key=" ",
        timeout=60.0
    )

def truncate_text(text, max_tokens):
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        return tokenizer.decode(tokens[:max_tokens])
    return text

async def process_batch(client, batch):

    messages_list = []
    for item in batch:
        issue_type = item["type"]
        summary = item["summary"]
        description = item["description"]
        
        truncated_summary = truncate_text(summary, MAX_INPUT_TOKENS // 3)
        truncated_description = truncate_text(description, MAX_INPUT_TOKENS * 2 // 3)
        
        system_prompt = (
            "You are a software engineer specializing in Jira issue analysis. "
            "Your task is to analyze and enrich the provided issue data."
        )
        
        user_prompt = (
            "**Issue Data:**\n"
            f"- **Type:** {issue_type}\n"
            f"- **Summary:** {truncated_summary}\n"
            f"- **Description:**\n{truncated_description}\n\n"
            "**Your Tasks:**\n"
            "1. **Summary**: Provide a one-sentence summary of the core issue.\n"
            "2. **Key Information**: Extract the following fields:\n"
            "   - Problem Description\n"
            "   - Reproduction Steps\n"
            "   - Expected Behavior\n"
            "   - Actual Behavior\n"
            "3. **Semantic Expansion**: List related technical concepts, components, or potentially affected modules.\n"
            "4. **User Intent Analysis**: Analyze the user's deepest need or purpose.\n\n"
            "**Output Format Requirements:**\n"
            "- Combine all information into a single coherent text paragraph.\n"
            "- Use clear section headings for each part.\n"
            "- Keep the entire response under 250 words.\n"
            "- Do not include any additional commentary outside the enriched description."
        )
        
        messages_list.append({
            "item": item,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        })
    
    # 发送批量请求
    responses = []
    for i in range(0, len(messages_list), BATCH_SIZE):
        batch_reqs = []
        for item in messages_list[i:i+BATCH_SIZE]:
            batch_reqs.append({
                "method": "chat.completions",
                "model": MODEL_NAME,
                "messages": item["messages"],
                "temperature": 0.3,
                "max_tokens": 450,
                "top_p": 0.9
            })
        
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=batch_reqs,
            )
            responses.extend(response.choices)
        except Exception:

            for req in batch_reqs:
                try:
                    resp = await client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=req["messages"],
                        temperature=0.3,
                        max_tokens=450,
                        top_p=0.9
                    )
                    responses.append(resp.choices[0])
                except Exception as e:
                    responses.append({
                        "error": str(e),
                        "item": req["item"]
                    })
    
    # 处理响应
    results = []
    for i, response in enumerate(responses):
        item = messages_list[i]["item"]
        if isinstance(response, dict) and "error" in response:
            # 错误处理
            fallback_text = f"Type: {item['type']}\nSummary: {item['summary']}\nError: {response['error']}"
            results.append({
                "project": item["project"],
                "key": item["key"],
                "augmented_text": fallback_text,
                "input_tokens": 0,
                "output_tokens": len(tokenizer.encode(fallback_text))
            })
            continue
        
        content = response.message.content.strip()
        input_tokens = len(tokenizer.encode(messages_list[i]["messages"][1]["content"]))
        output_tokens = len(tokenizer.encode(content))
        
        results.append({
            "project": item["project"],
            "key": item["key"],
            "augmented_text": content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        })
    
    return results

async def process_project(project_name):
    """处理单个项目"""
    print(f"\n{'='*60}")
    print(f" {project_name}")
    print(f"{'='*60}\n")
    
    # 构建文件路径
    input_csv = os.path.join(BASE_DATA_PATH, project_name, f"{project_name}-issue.csv")
    output_dir = os.path.join(BASE_DATA_PATH, project_name)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    output_csv = os.path.join(output_dir, f"{project_name}-augmented-issue.csv")
    checkpoint_csv = os.path.join(output_dir, f"{project_name}-checkpoint.csv")
    output_enriched_csv = os.path.join(output_dir, f"{project_name}-enriched-only.csv")
    
    print(f"INPUT: {input_csv}")
    print(f"OUTPUT: {output_csv}")
    print(f"CHECK: {checkpoint_csv}")
    

    try:
        df = pd.read_csv(input_csv, sep=';')
    except Exception as e:
        print(f" {project_name} failed: {e}")
        return

    df['Summary'] = df['Summary'].fillna('')
    df['DescriptionFull'] = df['DescriptionFull'].fillna('')
    df['Type'] = df['Type'].fillna('Unknown')

    df['Augmented_Text'] = ""
    df['Processing_Time'] = 0.0
    df['Input_Tokens'] = 0
    df['Output_Tokens'] = 0

    checkpoint = 0
    if os.path.exists(checkpoint_csv):
        try:
            checkpoint_df = pd.read_csv(checkpoint_csv)
            checkpoint = len(checkpoint_df)
            df = pd.concat([checkpoint_df, df.iloc[checkpoint:]], ignore_index=True)
            print(f"Resuming from checkpoint, {checkpoint} records have been processed.")
        except Exception as e:
            print(f"Checkpoint file is corrupted: {e}. Processing from the beginning.")
    
    total_count = len(df)
    if checkpoint >= total_count:
        print(f"All records for project {project_name} have been processed!")
        return
    
    start_time = time.time()
    total_tokens = 0
    client = get_async_openai_client()

    batch = []

    tasks = []
    for index in range(checkpoint, total_count):
        row = df.iloc[index]
        batch.append({
            "project": project_name,
            "key": row['Key'],
            "type": row['Type'],
            "summary": row['Summary'],
            "description": row['DescriptionFull'],
            "index": index
        })

        if len(batch) >= CONCURRENCY:
            tasks.append(process_batch(client, batch))
            batch = []

    if batch:
        tasks.append(process_batch(client, batch))

    completed = 0
    for task in asyncio.as_completed(tasks):
        batch_results = await task
        for result in batch_results:
            key = result["key"]
            matching_rows = df[df['Key'] == key]
            if not matching_rows.empty:
                index = matching_rows.index[0]
                df.at[index, 'Augmented_Text'] = result["augmented_text"]
                df.at[index, 'Input_Tokens'] = result["input_tokens"]
                df.at[index, 'Output_Tokens'] = result["output_tokens"]
                df.at[index, 'Processing_Time'] = time.time() - start_time
                
                total_tokens += (result["input_tokens"] + result["output_tokens"])
                completed += 1

                if completed % LOG_FREQUENCY == 0:
                    elapsed = time.time() - start_time
                    items_per_min = (completed / elapsed) * 60 if elapsed > 0 else 0
                    remaining = total_count - checkpoint - completed

                    eta_str = "未知"
                    if items_per_min > 0:
                        eta_minutes = remaining / items_per_min
                        eta_str = (datetime.now() + timedelta(minutes=eta_minutes)).strftime("%Y-%m-%d %H:%M:%S")

                    print(f"[{project_name}] 进度: {completed}/{total_count-checkpoint} | "
                          f"速度: {items_per_min:.1f} 条/分钟 | "
                          f"剩余: {remaining} | ETA: {eta_str}")

        df.to_csv(checkpoint_csv, index=False)

    df.to_csv(output_csv, index=False)

    simplified_df = df[['Key', 'Augmented_Text', 'Processing_Time', 'Input_Tokens', 'Output_Tokens']].copy()
    simplified_df.to_csv(output_enriched_csv, index=False)
    
    total_elapsed = time.time() - start_time
    items_per_min = (completed / total_elapsed) * 60 if total_elapsed > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"项目 {project_name} 处理完成!")
    print(f"总耗时: {total_elapsed:.2f}秒")
    print(f"处理速度: {items_per_min:.1f} 条/分钟")
    print(f"总Token消耗: {total_tokens}")
    print(f"结果保存到: {output_csv}")
    print(f"简化版保存到: {output_enriched_csv}")
    print(f"{'='*60}\n")

def process_all_projects():

    print(f"开始处理 {len(PROJECTS)} 个项目:")
    for project in PROJECTS:
        print(f"- {project}")

    # 顺序处理每个项目
    for project_name in PROJECTS:
        asyncio.run(process_project(project_name))
    
    print("All project tasks have been completed！")

if __name__ == "__main__":
    process_all_projects()