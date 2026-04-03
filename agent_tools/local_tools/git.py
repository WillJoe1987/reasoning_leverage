import os,sys
from pathlib import Path
from langchain.tools import tool
import subprocess
import fnmatch
import re
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def create_check_git_ignore_tool():
    @tool
    def is_git_ignored(path: str) -> dict:
        """
            判断指定文件或目录是否被 git 忽略。
        """
        try:
            p = Path(path)
            if not p.exists():
                return {"path": path, "ignored": False, "error": "path not exists"}
            # 找到最近的 git 根目录（包含 .git）
            cur = p if p.is_dir() else p.parent
            git_root = None
            while True:
                if (cur / ".git").exists():
                    git_root = cur
                    break
                if cur == cur.parent:
                    break
                cur = cur.parent

            relpath = None
            if git_root:
                try:
                    relpath = str(Path(path).resolve().relative_to(git_root.resolve()))
                except Exception:
                    relpath = str(Path(path).resolve())

            # 尝试使用 git check-ignore
            if git_root:
                try:
                    # 使用相对于 git_root 的路径以匹配 git 的行为
                    target = relpath if relpath is not None else str(Path(path))
                    res = subprocess.run(
                        ["git", "check-ignore", "-v", "--", target],
                        cwd=str(git_root),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    if res.returncode == 0:
                        # 输出格式通常为 "<gitignore-path>:<lineno>:<pattern>\t<path>"
                        out = res.stdout.strip()
                        # 提取 pattern 信息（尽量）
                        m = re.search(r":\d+:(.+)\t", out)
                        pattern = m.group(1).strip() if m else None
                        return {
                            "path": path,
                            "git_root": str(git_root),
                            "ignored": True,
                            "matched_pattern": pattern,
                            "method": "git",
                        }
                    else:
                        return {
                            "path": path,
                            "git_root": str(git_root),
                            "ignored": False,
                            "matched_pattern": None,
                            "method": "git",
                        }
                except FileNotFoundError:
                    # git 不可用，回退到解析 .gitignore
                    pass
                except Exception as e:
                    # 回退到解析，但记录错误
                    git_err = str(e)
            else:
                git_err = None

            # 回退：逐级读取 .gitignore（从 git_root 到 path 所在目录），构建模式列表
            # 如果没有 git_root，则以项目根 project_root 为起点查找 .gitignore
            search_root = Path(git_root) if git_root else Path(project_root)
            search_root = search_root.resolve()
            target_path = Path(path).resolve()
            try:
                rel_for_match = str(target_path.relative_to(search_root))
            except Exception:
                rel_for_match = str(target_path)

            patterns = []
            negations = []
            # 获取从 search_root 到目标目录（包含各层）的 .gitignore
            parts = [search_root] + list(search_root.joinpath('.').glob('..'))  # 保持结构，不直接用
            # 实际遍历：从 search_root 到目标 所有父目录
            cur = target_path if target_path.is_dir() else target_path.parent
            stack = []
            while True:
                stack.append(cur)
                if cur == search_root or cur == cur.parent:
                    break
                cur = cur.parent
            # 逆序：从 search_root 开始读取，后面的规则覆盖前面的
            stack = reversed(stack)
            for d in stack:
                gitignore_file = Path(d) / ".gitignore"
                if gitignore_file.exists():
                    try:
                        with open(gitignore_file, 'r') as f:
                            for line in f:
                                line = line.rstrip()
                                if not line or line.lstrip().startswith("#"):
                                    continue
                                if line.startswith("!"):
                                    negations.append(line[1:])
                                else:
                                    patterns.append(line)
                    except Exception:
                        continue

            matched_pattern = None
            ignored = False
            # 简单匹配逻辑：对每个 pattern 用 fnmatch 与相对路径匹配
            for pat in patterns:
                pat_strip = pat.strip()
                # 将 gitignore 中的目录样式 "foo/" 视为匹配以 foo/ 开头的路径
                if pat_strip.endswith('/'):
                    if rel_for_match.startswith(pat_strip.rstrip('/')) or fnmatch.fnmatch(rel_for_match, pat_strip + '*'):
                        matched_pattern = pat_strip
                        ignored = True
                else:
                    # 支持通配符和目录匹配
                    if fnmatch.fnmatch(rel_for_match, pat_strip) or fnmatch.fnmatch(os.path.basename(rel_for_match), pat_strip):
                        matched_pattern = pat_strip
                        ignored = True
            # 处理否定规则
            for pat in negations:
                if fnmatch.fnmatch(rel_for_match, pat) or fnmatch.fnmatch(os.path.basename(rel_for_match), pat) or rel_for_match.startswith(pat.rstrip('/')):
                    # 否定规则命中，取消忽略
                    matched_pattern = pat
                    ignored = False
                    break

            result = {
                "path": path,
                "git_root": str(git_root) if git_root else None,
                "ignored": ignored,
                "matched_pattern": matched_pattern,
                "method": "parse",
            }
            if git_err:
                result["git_error"] = git_err
            return result
        except Exception as e:
            return {"path": path, "ignored": False, "error": str(e)}
    
    return is_git_ignored


if __name__ == "__main__":

    is_git_ignored = create_check_git_ignore_tool()
    # 测试样例
    test_paths = [
        "/prog/pweb/AI-Trader/agent_tools/local_tools/git.py",
        "/prog/pweb/AI-Trader/.env",
        "/prog/pweb/AI-Trader/some_ignored_file.tmp",
        "/prog/pweb/AI-Trader/some_directory/",
    ]
    for tp in test_paths:
        result = is_git_ignored(tp)
        print(f"Path: {tp} => Ignored: {result['ignored']}, Pattern: {result.get('matched_pattern')}, Method: {result.get('method')}")