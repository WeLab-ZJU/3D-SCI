import os
import ast
from collections import defaultdict

def get_imports_from_file(filepath):
    """从单个Python文件中提取导入的库"""
    imports = set()
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
            
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # 获取顶级模块名
                    module_name = alias.name.split('.')[0]
                    imports.add(module_name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # 获取顶级模块名
                    module_name = node.module.split('.')[0]
                    imports.add(module_name)
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
    
    return imports

def scan_directory(directory):
    """扫描目录及其子目录中的所有Python文件"""
    all_imports = defaultdict(list)
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                imports = get_imports_from_file(filepath)
                for imp in imports:
                    all_imports[imp].append(filepath)
    
    return all_imports

if __name__ == "__main__":
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 扫描目录并获取所有导入
    imports = scan_directory(current_dir)
    
    # 打印结果
    print("Imported libraries and the files that use them:")
    print("=" * 60)
    for lib, files in sorted(imports.items()):
        print(f"\n{lib}:")
        for file in files:
            print(f"  - {file}")
    
    print("\n" + "=" * 60)
    print(f"Total unique libraries imported: {len(imports)}")
