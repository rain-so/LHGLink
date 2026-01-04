import csv
PROJECT_NAME = "cassandra"

issues = []
with open(f'../../dataset/issues/{PROJECT_NAME}-jira-issue-dataset.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f, delimiter=';')
    for row in reader:
        issues.append(row)

issue_index = {}
assignee_index = {}
component_index = {}

current_issue = 0
current_assignee = 0
current_component = 0

for issue in issues:
    key = issue['Key']
    if key not in issue_index:
        issue_index[key] = current_issue
        current_issue += 1

    assignee = issue['Assignee']
    if assignee and assignee not in assignee_index:
        assignee_index[assignee] = current_assignee
        current_assignee += 1

    components = issue['Components'].strip()
    if components:
        for comp in components.split():
            if comp not in component_index:
                component_index[comp] = current_component
                current_component += 1

def write_index(file_path, index_dict):
    with open(file_path, 'w', encoding='utf-8') as f:
        for key, idx in index_dict.items():
            f.write(f"{idx} {key}\n")


write_index(f'data/Index/issue_index.txt', issue_index)
write_index(f'data/Index/assignee_index.txt', assignee_index)
write_index(f'data/Index/component_index.txt', component_index)

# issue_assignee
with open(f'data/Index/issue_assignee_index.txt', 'w') as f:
    for issue in issues:
        key = issue['Key']
        assignee = issue['Assignee']
        if assignee:
            issue_idx = issue_index[key]
            assignee_idx = assignee_index[assignee]
            f.write(f"{issue_idx} {assignee_idx}\n")

# issue_component
with open(f'data/Index/issue_component_index.txt', 'w') as f:
    for issue in issues:
        key = issue['Key']
        components = issue['Components'].strip()
        if components:
            issue_idx = issue_index[key]
            for comp in components.split():
                comp_idx = component_index[comp]
                f.write(f"{issue_idx} {comp_idx}\n")

# issue_issue (处理Inward和Outward链接)
processed_pairs = set()

with open(f'data/Index/issue_issue_index.txt', 'w') as f:
    for issue in issues:
        current_key = issue['Key']
        current_idx = issue_index[current_key]

        # 统一处理所有链接（合并Outward和Inward）
        def process_links(links, is_outward=True):
            if links.strip():
                for link in links.split(','):
                    link = link.strip()
                    if link and ':' in link:
                        link_type, related_key = link.split(':', 1)
                        link_type = link_type.strip()
                        related_key = related_key.strip()

                        if related_key in issue_index:
                            # 确定source和target的索引
                            if is_outward:
                                source_idx = current_idx
                                target_idx = issue_index[related_key]
                            else:
                                source_idx = issue_index[related_key]
                                target_idx = current_idx

                            # 生成唯一标识符（有序对 + 链接类型）
                            identifier = tuple(sorted((source_idx, target_idx))) + (link_type,)

                            if identifier not in processed_pairs:
                                f.write(f"{source_idx} {target_idx} {link_type}\n")
                                processed_pairs.add(identifier)

        # 处理Outward链接
        process_links(issue['OutwardIssueLinks'], is_outward=True)

        # 处理Inward链接（转换为Outward方向）
        process_links(issue['InwardIssueLinks'], is_outward=False)