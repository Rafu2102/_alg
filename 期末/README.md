# 作業總覽

---

## 習題1：計算合併排序的複雜度 T(n) = T(n/2) + n

[hw1](https://github.com/Rafu2102/_sp/tree/main/hw1)

- **原始程式**：老師提供的簡易編譯器  
- **完成方式**：  
  - **原創**：手寫 `DOWHILE()`、修改 `STMT()`  
  - **參考**：課堂教材；ChatGPT 用於語法設計參考  
- **原理**：  
  1. 在 Parser 中加入 `do` → `DOWHILE()` 支援  
  2. `skip("do")`…`skip("while")`…`E()` 解析條件  
  3. emit 兩組 label 形成迴圈  
- **使用方法**：  
  1. 編譯編譯器  
  2. 在輸入檔案中撰寫  
     ```C
     do { print(x); x = x-1; } while (x>0);
     ```  
  3. 觀察輸出中間碼是否符合預期  

---

---

## 習題 2：C4 組合語言理解與硬塞練習

[hw2](https://github.com/Rafu2102/_sp/tree/main/hw2)

- **原始程式**：`power2.c`  
- **完成方式**：  
  - **原創**：手動撰寫 power2.c，使用迴圈累乘實作   
  - **參考**：利用 c4.c 提供的 VM，將 power2.c 編譯成自訂組合語；ChatGPT 用於語法設計參考 
  - **硬塞**：將產生的 opcode 陣列硬塞進新的 C 程式 (c4_power2_embedded.c)，以 run() 延後執行。
- **原理**：  
  1. `power2.c` 先用迴圈累乘或位移計算 \(2^n\)  
  2. 使用 `c4 power2.c -o power2.s` 產生組合語  
  3. 將 `power2.s` 中的 opcode 陣列硬塞到新的 C 檔案，延後由 VM 執行  
- **使用方法**：  
  1. 編譯並執行 `power2.c`：  
  2. 產生組合語：  
     ```
     c4 power2.c -o power2.s
     ```  
     檢視 `power2.s` 中的 `IMM`, `MUL`, `BZ`, `JMP`, `PRTF` 等指令  
  3. 撰寫 `c4_power2_embedded.c`，將 opcode 硬塞為 `int code[] = { ... };`  
  4. 編譯虛擬機執行：  
     ```bash
     gcc -std=c99 c4_power2_embedded.c -o c4_emb -lm
     ./c4_emb
     ```  

---

---

## 習題 3：為 C4 編譯器加入 `do…while`

[hw3](https://github.com/Rafu2102/_sp/tree/main/hw3)

- **原始程式**：老師的 C4 編譯器源碼  
- **完成方式**：  
  - **原創**：參考老師的編譯器在詞法分析與程式生成階段加入對 `do…while` 的支援  
  - **參考**：習題 1 中 `DOWHILE()` 實作思路  ；ChatGPT 用於語法設計參考 
- **原理**：  
  1. 在詞法分析階段，`next()` 已將 `do` 解析為 `Do` token；  
  2. 在 `stmt()` 中加入：  
     ```c
     else if (tk == Do) {
       next();                           // 讀過 do
       a = e + 1;                        // 記錄迴圈開始位址
       stmt();                           // 先執行一次迴圈主體
       next(); /* expect '(' */         
       expr(Assign);                     // 解析 while 的判斷式
       next(); /* expect ')' */          
       next(); /* expect ';' */          
       *++e = BZ; b = ++e;               // 如果條件不成立，跳過迴圈
       *++e = JMP; *++e = (int)a;        // 否則跳回迴圈開始
       *b = (int)(e + 1);                // 填入迴圈結束標號
     }
     ```  
     如此可確保迴圈體至少執行一次，再根據條件決定是否重複執行。  
- **使用方法**：  
  1. 編譯修改後的 C4 編譯器：  
     ```bash
     gcc -std=c99 c4.c -o c4
     ```  
  2. 建立測試檔 `test.c`，內容示例：:contentReference[oaicite:5]{index=5}  
     ```c
     #include <stdio.h>
     int power2(int n) {
         int r = 1;
         do {
             r = r * 2;
             n = n - 1;
         } while (n > 0);
         return r;
     }
     int main() {
         int arr_size = 4, i = 0, res = power2(3);
         printf("power2(3) = %d\n", res);
         int *arr = malloc(arr_size * sizeof(int));
         do {
             arr[i] = (i + 1) * res;
             i = i + 1;
         } while (i < arr_size);
         i = 0;
         while (i < arr_size) {
             printf("arr[%d] = %d\n", i, arr[i]);
             i = i + 1;
         }
         free(arr);
         return 0;
     }
     ```  
  3. 執行並驗證：  
     ```bash
     ./c4 test.c
     ```  
     **預期輸出**：  
     ```
     power2(3) = 8
     arr[0] = 8
     arr[1] = 16
     arr[2] = 24
     arr[3] = 32
     exit(0) cycle = 401
     ``` :contentReference[oaicite:6]{index=6}  

---

---

## 習題 4：組合語言計算三數連乘並印出結果

[hw4](https://github.com/Rafu2102/_sp/tree/main/hw4)

- **原始程式**：  
  - C 呼叫範例 `mul3.c`，定義 `int mul3(int a,int b,int c)`，在 `main()` 中呼叫並 `printf` 輸出 :contentReference[oaicite:6]{index=6}  
  - 32 位元組合語 `mul3.s`，實作 Windows x64 呼叫約定下的 `mul3` 函式 :contentReference[oaicite:7]{index=7}  

- **完成方式**：  
  - **原創**：手動撰寫 `mul3.s`（32-bit 版本）與 `mul3-64bit.s`（64-bit 版本），並撰寫對應的 C 呼叫程式。  
  - **參考**：系統程式教材中的函式呼叫約定；ChatGPT 用於排版與說明優化。  

- **原理**：  
  1. **Windows x64 呼叫約定**：前三個 32-bit 參數依序放在 `ecx`, `edx`, `r8d`；32-bit 回傳在 `eax` :contentReference[oaicite:8]{index=8}  
  2. **32-bit 版本**：  
     ```asm
     mul3:
         mov   %ecx, %eax     ; eax = a
         imul  %edx, %eax     ; eax *= b
         imul  %r8d, %eax     ; eax *= c
         ret
     ```  
  3. **64-bit 版本**（避免 32-bit 溢出）：使用 `movq`、`imulq` 操作 64-bit 寄存器 (`rcx`,`rdx`,`r8`,`rax`) :contentReference[oaicite:9]{index=9}  

- **使用方法**：  
  1. **32-bit 版本**  
     ```bash
     gcc -std=c99 mul3.c mul3.s -o mul3
     ./mul3       # 輸出: mul3(3,2,5)=30
     ```  
  2. **64-bit 版本**  
     ```bash
     gcc -std=c99 mul3-64bit.c mul3-64bit.s -o mul3-64bit
     ./mul3-64bit # 輸出: mul3(55555,9487666,5201314)=2741546472768003820
     ```  

- **結果與分析**：  
  - **32-bit 溢出**：對大數相乘時結果變負，因 32-bit `movl`/`imull` 溢出 :contentReference[oaicite:10]{index=10}  
  - **64-bit 修正**：將 C 函式參數與回傳改為 `long long`，並在組合語中使用 `movq`/`imulq`，成功計算大數乘積。  

---

---

## 習題 5：C 程式與 objdump 分析

[hw5](https://github.com/Rafu2102/_sp/tree/main/hw5)

- **原始程式**：`factorial.c` 實作遞迴計算階乘  
  ```c
  // factorial.c 
  #include <stdio.h>

  // 計算 n 的階乘
  int factorial(int n) {
      if (n <= 1) {
          return 1;
      }
      return n * factorial(n - 1);
  }

  int main() {
      int n = 5;
      int result = factorial(n);
      printf("factorial(%d) = %d\n", n, result);
      return 0;
  }
    ```

  - **完成方式**：  
  - **原創**：手寫遞迴 factorial 函式並撰寫 main 呼叫 
  - **參考**：ChatGPT 用於排版；系統程式教材指出 gcc/objdump 使用方法

  - **原理**：  
  1. 使用 gcc -S 將 C 程式翻譯成組合語（.s）
  2. 使用 gcc -c 編譯成目標檔（.o）
  3. 使用 objdump -d 反組譯查看機器指令
  4. 使用 objdump -h 列印頭標（section headers）

- **使用方法**：  
  1. **產生組合語**  
     ```bash
     gcc -O0 -std=c99 -S factorial.c -o factorial.s
     ```  
  2. **編譯成目標檔**  
     ```bash
     gcc -std=c99 -c factorial.c -o factorial.o
     ```  
  3. **反組譯**  
     ```bash
     objdump -d factorial.o > factorial_dis.txt
     ```  
  4. **列印頭鏢**  
     ```bash
     objdump -h factorial.o > factorial_hdr.txt
     ```  
     - **結果與分析**：  
  - **執行結果**：factorial(5) = 120。  
  - **64-bit 修正**：使用 unsigned long long，遞迴計算 20!，避免 32 位元溢位。  

  ---

  ## 期中作業：請寫一個系統程式相關的專案或報告

  [算域虛擬機](https://github.com/Rafu2102/_sp/tree/main/%E6%9C%9F%E4%B8%AD/ucalc-vm)