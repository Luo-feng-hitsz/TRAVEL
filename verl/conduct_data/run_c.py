# import os
# import tempfile
# import subprocess
# from typing import Any, List, Dict, Optional
# import jsonlines
# import json
# import re


# def _which(cmd: str) -> bool:
#     return subprocess.run(["bash", "-lc", f"command -v {cmd} >/dev/null 2>&1"]).returncode == 0


# def _pick_compiler() -> str:
#     for cc in ("gcc", "clang"):
#         if _which(cc):
#             return cc
#     raise RuntimeError("No C compiler found. Please install gcc or clang.")


# def compile_c_code(c_code: str, workdir: str, extra_cflags: Optional[List[str]] = None) -> str:
#     """
#     Write main.c and compile to main.out, returning executable path.
#     """
#     if extra_cflags is None:
#         extra_cflags = []

#     c_path = os.path.join(workdir, "main.c")
#     exe_path = os.path.join(workdir, "main.out")

#     with open(c_path, "w", encoding="utf-8") as f:
#         f.write(c_code)

#     cc = _pick_compiler()
#     base = [cc, "-O2", "-std=c11", "-pipe", *extra_cflags, c_path, "-o", exe_path]

#     # try static first (optional), then fallback
#     attempts = [
#         [cc, "-O2", "-std=c11", "-pipe", "-static", *extra_cflags, c_path, "-o", exe_path],
#         base,
#     ]

#     last = None
#     for cmd in attempts:
#         r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         if r.returncode == 0 and os.path.exists(exe_path):
#             return exe_path
#         last = (cmd, r.stdout.decode("utf-8", "replace"), r.stderr.decode("utf-8", "replace"))

#     cmd, out_s, err_s = last
#     raise RuntimeError(
#         "C compilation failed.\n"
#         f"Command: {' '.join(cmd)}\n"
#         f"stdout:\n{out_s}\n"
#         f"stderr:\n{err_s}\n"
#     )


# def testcase_to_stdin(tc: List[Any]) -> str:
#     """
#     Match your C input:
#       scanf("%d",&t);
#       while(t--) scanf("%s", str);

#     testcase format: [t, s1, s2, ...]
#     """
#     if not isinstance(tc, list) or len(tc) < 2:
#         raise ValueError(f"Bad testcase, need [t, s1, ...]: {tc}")

#     t = tc[0]
#     strs = tc[1:]

#     if not isinstance(t, int):
#         raise ValueError(f"t must be int, got {type(t)}: {t}")
#     if len(strs) < t:
#         raise ValueError(f"testcase expects {t} strings, but got {len(strs)}: {tc}")

#     lines = [str(t)]
#     for s in strs[:t]:
#         s = str(s)
#         if any(ch.isspace() for ch in s):
#             raise ValueError(f"String contains whitespace; scanf('%s') can't read it: {repr(s)}")
#         lines.append(s)

#     return "\n".join(lines) + "\n"


# def run_exe(exe_path: str, stdin_data: str, timeout_sec: float = 2.0) -> Dict[str, Any]:
#     r = subprocess.run(
#         [exe_path],
#         input=stdin_data.encode("utf-8"),
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#         timeout=timeout_sec,
#     )
#     return {
#         "stdin": stdin_data,
#         "stdout": r.stdout.decode("utf-8", "replace"),
#         "stderr": r.stderr.decode("utf-8", "replace"),
#         "returncode": r.returncode,
#     }


# def run_c_on_testcases(
#     c_code: str,
#     testcases: List[List[Any]],
#     timeout_sec: float = 2.0,
#     extra_cflags: Optional[List[str]] = None,
# ) -> List[Dict[str, Any]]:
#     """
#     Compile once, run all testcases.
#     Returns list of dict per testcase: stdin/stdout/stderr/returncode
#     """
#     with tempfile.TemporaryDirectory(prefix="c_runner_") as workdir:
#         exe_path = compile_c_code(c_code, workdir, extra_cflags=extra_cflags)

#         results = []
#         for idx, tc in enumerate(testcases):
#             stdin_data = testcase_to_stdin(tc)
#             res = run_exe(exe_path, stdin_data, timeout_sec=timeout_sec)
#             res["case_index"] = idx
#             results.append(res)
#         return results

# # ---------------- Example usage ----------------
# if __name__ == "__main__":
#     c_code = r'''
# #include<stdio.h>
# #include<stdlib.h>
# int main()
# {
#     int n, *a, m, x, y, i;
#     scanf("%d",&n);
#     a = (int*)malloc(n*sizeof(int));
#     for(i = 0; i < n; i++)
#         scanf("%d",&a[i]);
#     scanf("%d", &m);
#     for(i = 1; i <= m; i++ )
#     {
#         scanf("%d%d",&x, &y);
#         if(((x-1) != 0) || (x != n))
#         {
#             a[x-2] += y-1;
#             a[x] += a[x-1]-y;
#         }
#         a[x-1] = 0;
#     }
#     for(i =0; i < n; i++)
#         printf("%d\n",a[i]);
#     return 0;
# }
# '''
#     testcases_gpt = [
# [1, 10, 1, 1, 5],
# [5, 1, 2, 3, 4, 5, 2, 3, 2, 2, 1],
# [3, 0, 5, 10, 1, 2, 5],
# [4, 7, 0, 3, 8, 3, 2, 2, 3, 1, 2, 3],
# [2, 9, 4, 0]
# ]
#     testcases_qwen = [[3, 1, 2, 3, 2, 1, 2], [5, 10, 20, 30, 40, 50, 3, 1, 5, 3, 10, 5, 15], [1, 100, 1, 1, 1], [4, 0, 0, 0, 0, 2, 2, 1, 3, 1, 1], [6, 1, 1, 1, 1, 1, 1, 4, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6]]

#     results_gpt = run_c_on_testcases(c_code, testcases_gpt, timeout_sec=2.0)
#     outputs = [r["stdout"] for r in results_gpt]
#     print("GPT outputs:")
#     print(outputs)
#     results_qwen = run_c_on_testcases(c_code, testcases_qwen, timeout_sec=2.0)
#     outputs = [r["stdout"] for r in results_qwen]
#     print(outputs)

import subprocess
import tempfile
import os

def run_c_on_testcases(c_code: str, inputs: list, timeout: float = 2.0):
    """
    timeout: 每个测试用例最大运行时间（秒）
    """
    outputs = []

    # 1️⃣ 写入临时 C 文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".c") as f:
        c_path = f.name
        f.write(c_code.encode())

    exe_path = c_path[:-2]

    try:
        # 2️⃣ 编译
        compile_result = subprocess.run(
            ["gcc", c_path, "-O2", "-std=c11", "-o", exe_path],
            capture_output=True,
            text=True
        )

        if compile_result.returncode != 0:
            raise RuntimeError("编译失败:\n" + compile_result.stderr)

        # 3️⃣ 逐个输入运行
        for value in inputs:
            all_input = "\n".join(map(str, value)) + "\n"

            try:
                result = subprocess.run(
                    [exe_path],
                    input=all_input,
                    capture_output=True,
                    text=True,
                    timeout=timeout   # ⭐ 超时控制
                )

                outputs.append({
                    "status": "OK",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                })

            except subprocess.TimeoutExpired as e:
                outputs.append({
                    "status": "TIMEOUT",
                    "stdout": e.stdout,
                    "stderr": e.stderr,
                    "returncode": None
                })

        return outputs

    finally:
        # 4️⃣ 清理文件
        if os.path.exists(c_path):
            os.remove(c_path)
        if os.path.exists(exe_path):
            os.remove(exe_path)

if __name__ == "__main__":
    c_code = r'''
    #include<stdio.h>
    char square[1001];
    int main(void)
    {
            int t;
            scanf("%d", &t);
            int n, m, x, y;
            while (t--)
            {
                    int Wpointsum = 0, DWpointsum = 0;
                    scanf("%d %d %d %d", &n, &m, &x, &y);
                    while(n--)
                    {
                            scanf("%s", square, m + 1);
                            for (int i = 0; i < m; i++)
                            {
                                    if (square[i] == '.')
                                    {
                                            if (square[i + 1] != '.')
                                                    Wpointsum++;
                                            else
                                            {
                                                    DWpointsum++;
                                                    i++;
                                            }
                                    }
                            }
                    }
                    if (2 * x <= y)
                            printf("%d\n", Wpointsum * x + DWpointsum * 2 * x);
                    else
                            printf("%d\n", Wpointsum * x + DWpointsum * y);
            }
            return 0;
    }
    '''
    testcases_qwen = [[1, 3, 3, 1, 1, 1, '...', '...', '...'], [2, 4, 4, 2, 2, 2, '....', '....', '....', '....', '....', '....', '....', '....'], [1, 5, 5, 3, 3, 3, '.....', '.....', '.....', '.....', '.....'], [1, 6, 6, 4, 4, 4, '......', '......', '......', '......', '......', '......'], [1, 7, 7, 5, 5, 5, '.......', '.......', '.......', '.......', '.......', '.......', '.......']]
    result = run_c_on_testcases(c_code, testcases_qwen)