import subprocess
import tempfile
import os

def run_c_on_testcases(c_code: str, inputs: list):
    outputs = []

    # 1️⃣ 写入临时 C 文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".c") as f:
        c_path = f.name
        f.write(c_code.encode())

    exe_path = c_path[:-2]  # 去掉 .c

    try:
        # 2️⃣ 编译
        compile_result = subprocess.run(
            ["gcc", c_path, "-o", exe_path],
            capture_output=True,
            text=True
        )

        if compile_result.returncode != 0:
            raise RuntimeError("编译失败:\n" + compile_result.stderr)

        # 3️⃣ 逐个输入运行
        outputs = []
        for value in inputs:
            all_input = "\n".join(map(str, value)) + "\n"
            # print(f"Running with input:\n{all_input}")
            result = subprocess.run(
                [exe_path],
                input=all_input,
                capture_output=True,
                text=True
            )
            # print(f"Output:\n{result.stdout}")
            result = [int(line) for line in result.stdout.strip().split('\n') if line.strip()]
            outputs.append(result)

        return outputs

    finally:
        # 4️⃣ 清理文件
        os.remove(c_path)
        if os.path.exists(exe_path):
            os.remove(exe_path)

if __name__ == "__main__":
    c_code = r'''
    #include<stdio.h>
    #include<stdlib.h>
    int main()
    {
        int n, *a, m, x, y, i;
        scanf("%d",&n);
        a = (int*)malloc(n*sizeof(int));
        for(i = 0; i < n; i++)
            scanf("%d",&a[i]);
        scanf("%d", &m);
        for(i = 1; i <= m; i++ )
        {
            scanf("%d%d",&x, &y);
            if(((x-1) != 0) || (x != n))
            {
                a[x-2] += y-1;
                a[x] += a[x-1]-y;
            }
            a[x-1] = 0;
        }
        for(i =0; i < n; i++)
            printf("%d\n",a[i]);
        return 0;
    }
    '''
    testcases_qwen = [[3, 1, 2, 3, 2, 1, 2], [5, 10, 20, 30, 40, 50, 3, 1, 5, 3, 10, 5, 15], [1, 100, 1, 1, 1], [4, 0, 0, 0, 0, 2, 2, 1, 3, 1, 1], [6, 1, 1, 1, 1, 1, 1, 4, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6]]
    result = run_c_on_testcases(c_code, testcases_qwen)

    print(result)