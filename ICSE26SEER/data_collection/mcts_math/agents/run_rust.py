import subprocess
import tempfile
import os
import sys

def run_rust_on_testcases(rust_code: str, inputs: list):
    outputs = []

    # 1️⃣ 写入临时 Rust 文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".rs") as f:
        rs_path = f.name
        f.write(rust_code.encode("utf-8"))

    # 生成可执行文件路径（Windows 需要 .exe）
    exe_path = rs_path[:-3] + (".exe" if os.name == "nt" else "")

    try:
        # 2️⃣ 编译 Rust
        # -O: 优化（可选）
        compile_result = subprocess.run(
            ["rustc", rs_path, "-O", "-o", exe_path],
            capture_output=True,
            text=True
        )

        if compile_result.returncode != 0:
            raise RuntimeError("编译失败:\n" + (compile_result.stderr or compile_result.stdout))

        # 3️⃣ 逐个输入运行
        outputs = []
        for value in inputs:
            all_input = "\n".join(map(str, value)) + "\n"
            result = subprocess.run(
                [exe_path],
                input=all_input,
                capture_output=True,
                text=True
            )

            # 如果你希望运行时错误也抛出（可选）
            if result.returncode != 0:
                raise RuntimeError("运行失败:\n" + (result.stderr or result.stdout))

            # 仿照你原来的解析：按行解析成 int 列表
            parsed = [int(line) for line in result.stdout.strip().split("\n") if line.strip()]
            outputs.append(parsed)

        return outputs

    finally:
        # 4️⃣ 清理文件
        try:
            os.remove(rs_path)
        except FileNotFoundError:
            pass
        if os.path.exists(exe_path):
            try:
                os.remove(exe_path)
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    rust_code = r'''
    use std::io::{self, Read};

    fn main() {
        // 读入全部 stdin
        let mut input = String::new();
        io::stdin().read_to_string(&mut input).unwrap();
        let mut it = input.split_whitespace();

        let n: usize = match it.next() {
            Some(x) => x.parse().unwrap(),
            None => return,
        };

        let mut a = vec![0i64; n];
        for i in 0..n {
            a[i] = it.next().unwrap().parse().unwrap();
        }

        let m: usize = it.next().unwrap().parse().unwrap();
        for _ in 0..m {
            let x: i64 = it.next().unwrap().parse().unwrap();
            let y: i64 = it.next().unwrap().parse().unwrap();

            // 下面只是把你 C 代码逻辑“照搬”到 Rust 的一种写法（注意边界）
            // 原 C 代码有越界风险，这里做了些边界保护，避免 panic。
            let xi = x as isize; // 1-based
            if xi - 2 >= 0 && (xi - 2) < n as isize {
                a[(xi - 2) as usize] += y - 1;
            }
            if xi < n as isize {
                // a[x] += a[x-1]-y
                // x-1 对应 xi-1
                if xi - 1 >= 0 && (xi - 1) < n as isize {
                    let prev = a[(xi - 1) as usize];
                    a[xi as usize] += prev - y;
                }
            }
            if xi - 1 >= 0 && (xi - 1) < n as isize {
                a[(xi - 1) as usize] = 0;
            }
        }

        for v in a {
            println!("{}", v);
        }
    }
    '''

    testcases_qwen = [
        [3, 1, 2, 3, 2, 1, 2],
        [5, 10, 20, 30, 40, 50, 3, 1, 5, 3, 10, 5, 15],
        [1, 100, 1, 1, 1],
        [4, 0, 0, 0, 0, 2, 2, 1, 3, 1, 1],
        [6, 1, 1, 1, 1, 1, 1, 4, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
    ]

    result = run_rust_on_testcases(rust_code, testcases_qwen)
    print(result)