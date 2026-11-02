# MicroRing Studio（开源光子芯片 PIC、微环谐振器、MZI 调制器、端面/光栅耦合器与超高 Q 仿真设计平台）

**无需 Ansys Lumerical MODE/FDTD、Synopsys RSoft 或 COMSOL 商业许可证 —— 通过中英双语 AI 光子设计助手（Copilot）、2D 全矢量本征模与空间耦合模引擎、零 3D-FDTD 马赫-曾德尔干涉仪（`MZI`：`Y 分支` / `DC` / `MMI`）与高速调制器分析、纯自研 2D-FEM 端面耦合器（Butt/Edge Coupler）与 2D 全波 `FDFD` 表面光栅耦合器（Grating Coupler）、超高 Q（`10^4`–`10^8+`）瞬时稳态收敛引擎、2D-FEM 有源调谐（`TO`/`EO`/`硅 PN`）、Lumerical INTERCONNECT 导出及双向 GDSII 版图桥，一站式打通光子集成电路（PIC）与光模块互连从物理仿真到流片掩膜的全流程。**

[English](../README.md) | [简体中文](./README_zh-CN.md) | [求解器交叉验证与基准](./VERIFICATION_AND_BENCHMARKS.md) | [对话与 GDS 进阶教程](./tutorials/chatbot_and_gds_guide.md) | [Codex 与 Claude 极省 Token 接入指南](./tutorials/ai_agents_codex_claude.md) | [llms.txt](./llms.txt) | [更新日志](../CHANGELOG.md)

[![Run Online in Browser](https://img.shields.io/badge/Run_Online-GitHub_Codespaces-2ea44f?logo=github)](https://codespaces.new/ZhihaoRenKevin/microring-studio?quickstart=1)
[![Desktop Installers: .dmg & .exe](https://img.shields.io/badge/Desktop_App-macOS_.dmg_%7C_Win_.exe-0284c7)](https://github.com/ZhihaoRenKevin/microring-studio/releases/latest)
[![Release: v3.6.0](https://img.shields.io/badge/release-v3.6.0-10b981.svg)](https://github.com/ZhihaoRenKevin/microring-studio/releases/tag/v3.6.0)
[![Wavebands: 400nm–14µm](https://img.shields.io/badge/wavebands-400nm%20Vis%20to%2014%CE%BCm%20Mid--IR-ec4899.svg)](./VERIFICATION_AND_BENCHMARKS.md)
[![AI Modes: Offline | Claude | Codex | Ollama](https://img.shields.io/badge/AI_Modes-Offline_%7C_Claude_%7C_Codex_%7C_Ollama-8b5cf6.svg)](./tutorials/ai_agents_codex_claude.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](../LICENSE)
[![Python 3.11 | 3.12 | 3.13](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](../pyproject.toml)

![MicroRing Studio Overview](./images/hero_microring_overview.png)
*上图完全由仓库内 `benchmarks/generate_hero_figure.py` 无商业软件直接求解生成：**（a）** 2D 全矢量弯曲波导保形有限元模场（`4H-SiC`，`R = 40 um`，`82°` 侧壁倾角），**（b）** 同心滑轮耦合器（Pulley）包角扫描与自动临界耦合点（`theta_CC`）综合，**（c）** `X切薄膜铌酸锂（TFLN）` 跑道微环电光调谐光谱（`0 V` vs `+25 V`），**（d）** 实测级顺/逆时针背向散射劈裂双峰（Split-Doublet）CMT 参数提取。*

---

## 核心亮点

1. **零 3D-FDTD 马赫-曾德尔干涉仪（`MZI`）与高速电光/热光调制器套件（`v3.6.0` 新增）**：
   - **可选 4 类 `50:50` 分束/合束器，全部由原生 2D 保形全矢量 FEM + 多模本征模展开（`EME`）直接求解**：**`2x2 DC` 定向耦合器**采用严格镜像对称双芯奇偶超模色散（$n_{\mathrm{even}}, n_{\mathrm{odd}}$，插损 `0.044 dB`）；**`1x2` / `2x2 MMI` 多模干涉仪**采用多阶导模 2D-FEM `EME` + 复坡印廷锥形波导重叠积分 + 自成像长度自动寻优（输出二维 `|E(x,z)|` 干涉自成像场图）；**`1x2 Y 分支`**采用分叉缩腰尖端偶超模重叠积分（`130 nm` 工艺最小间距下插损 `0.273 dB`）。
   - **完整调制器指标、微环辅助 `RAMZI` 与 INTERCONNECT / GDS 导出**：支持平衡/非平衡臂（给定 $\Delta L$ 或目标 `FSR`）、微环辅助线性化 `RAMZI`，以及推挽（`push-pull`）/ 单臂驱动下的 `硅 PN 耗尽`、`X切 TFLN / SiC / ScAlN Pockels 电光`、`TiN 热光` 与 `硅 PIN 注入` 调制分析（直接输出 $V_\pi L$、$V_\pi$、$P_\pi$、非谐振 RC/热学 `3-dB` 带宽、开关传递曲线 $T(u)$、流片 `.GDS` 版图与 **Ansys Lumerical INTERCONNECT** `SPAR + OMM + .lsf` 模型包）。
2. **解决高 Q 微环在 Lumerical FDTD 中极慢且无法收敛的痛点（`v3.5.0`）**：
   - 对于低损耗 `Si3N4`、`TFLN` 或 `4H-SiC` 高 Q 微环（如 `Q_loaded = 3.02 × 10^6`，腔内光子寿命 `τ_ph = 2487 ps`），时域 3D-FDTD 要达到 `auto shutoff = 1e-5` 需仿真 **`28.64 ns`（超过 `8,200` 圈往返！）**；提前截断会导致严重的 FFT `sinc` 振铃伪影并使表观消光比从 `64.8 dB` 坍塌至 `< 2 dB`。
   - **MicroRing Studio** 采用亚飞米级（`0.1 fm`）双曲正弦谐振加密与解析无限圈稳态传递矩阵（$M \to \infty$），在 **`~1 秒`** 内精确求解任意 `Q = 10^4` 至 `10^8+` 谐振谱，并支持一键导出 **单程耦合区 3D-FDTD 切片（`.GDS` + `.lsf`，仅需 `0.67 ps`，提速 `> 40,000×`）**。
2. **100% 免商业软件的芯片端面耦合器（Butt/Edge Coupler）与二维全波 `FDFD` 表面光栅耦合器（Grating Coupler，`v3.5.0` 新增）**：
   - **端面模斑转换器（`design_butt_edge_coupler`）**：基于原生 2D 保形全矢量 Maxwell FEM 与复数坡印廷矢量重叠积分，自动优选倒锥尖端宽度 `W_tip`，精准对接 8 类光纤与 III-V DFB 半导体激光器（如 `2.5 µm` 透镜光纤总插损 **`0.56 dB`**、`2.4 × 1.2 µm` DFB 激光器直耦 **`0.61 dB`**），给出 `1-dB / 3-dB` 横纵对准容差与绝热锥长。
   - **表面光栅耦合器（`design_grating_coupler`）**：二阶 Rytov 等效介质 Bragg / 高斯切趾齿列综合 + 原生 **二维全波频域有限差分（`2D Full-Wave Maxwell FDFD` + `UPML` 吸收边界）**，完整计算浅刻蚀齿槽、上包层、`BOX` 埋氧层与底层硅衬底反射干涉（`220 nm` SOI 切趾聚焦光栅峰值插损 **`2.62 dB`**，`1-dB` 带宽 **`34.8 nm`**，约 `2.5 秒` 出全谱与二维辐射电场图），支持一键导出 **`127 µm` 光纤阵列 + 微环回环校准 `.GDS`** 与 **INTERCONNECT 2 端口 `SPAR` 压缩包**。
3. **6 步引导式中英双语光子设计助手（`EN | 中文` 一键切换）**：
   告别复杂的参数面板，每次只问一个清晰问题（`1. 选择目标` $\rightarrow$ `2. 工作波段` $\rightarrow$ `3. 波导材料` $\rightarrow$ `4. 晶圆截面与单模截止` $\rightarrow$ `5. 传播损耗 dB/cm` $\rightarrow$ `6. 微环拓扑、目标 FSR、临界耦合与 GDS 导出`），顶部可随时一键切换中英文。
2. **纯自研 2D-FEM 多物理场有源调谐（`TO` / `EO` / `硅 PN-PIN`）与 Ansys Lumerical INTERCONNECT 导出（`v3.4.0` 新增，零 COMSOL 依赖）**：
   - **截面多物理场有限元全自研**：内置 **2D Fourier 热传导有限元**（关联微加热器材质/宽度/间距与埋氧层 `BOX` 厚度，直接给出 $R_{\mathrm{th}}$、$P_\pi$ 与 $f_{3\mathrm{dB,TO}}$）、**2D 各向异性 Laplace 静电场有限元**（关联共面 `GSG` 或垂直电极间距与介电屏蔽，给出 $V_\pi L$ 与电光电容带宽）以及 **2D 硅横向 PN 耗尽 / PIN 注入模型**（Soref-Bennett 复数等离子体色散 $\Delta n + i\Delta k$，同步捕获谐振漂移与自由载流子吸收 FCA）。
   - **支持 `电压 V / 电流 mA / 功率 mW / 直接温升 ΔT [K] / 直接电场 E [V/µm]` 多态光谱与 INTERCONNECT 紧凑模型一键打包**：将二维复折射率分布实时送入 2D 保形全矢量 Maxwell 本征模求解器，并一键导出经过无源性与单调频率校验的 **Lumerical INTERCONNECT** 压缩包（`SPAR .dat`、`OMM n_eff .txt`、自动建图 `.lsf` 与 `lumapi` 脚本）。
3. **洁净室光刻过曝/欠曝（`正胶 Positive PR` vs `负胶 Negative PR`）与 CD 工艺容差引擎**：
   - **真实正负胶物理耦合**：内置 **正胶（ZEP520A / PMMA / KrF / ArF）**（过曝使波导宽度 $W$ 变窄、耦合间距 $\mathrm{gap}$ 变宽 $\rightarrow$ 欠耦合且谐振蓝移）与 **负胶（HSQ / ma-N 2403 / SU-8）**（过曝使波导展宽、间距收缩 $\rightarrow$ 过耦合、谐振红移及 `< 80 nm` 粘连/RIE lag 告警）全矢量 FEM + 空间 CMT 联立评估。
   - **良率窗口与一键导出 5 单元预偏置补偿版图（Pre-Biased `.GDS`）**：自动给出 `ER >= 15 dB` 剂量/线宽容差窗口、谐振漂移灵敏度 $d\lambda_0/d\mathrm{CD}$（`pm/nm`）、TiN 热调谐补偿温升 $\Delta T$，并一键导出 `[-20, -10, 0, +10, +20] nm` 五档光刻预补偿微环阵列 `.gds` + KLayout `.lyp`。
3. **双向波导传播损耗（`dB/cm`）设计与实验光谱反演**：
   - **已知损耗正向设计**：直接输入工艺线或文献波导传播损耗 $\alpha$（`dB/cm`）或 $Q_{\mathrm{int}}$，自动换算临界耦合所需精确间距 `gap`、滑轮包角 $\theta_{\mathrm{CC}}$ 或跑道直耦合长度 $L_{\mathrm{CC}}$，并支持损耗容差扫描。
   - **实测光谱损耗提取 $\rightarrow$ 闭环设计**：拖入实验透射谱 `.CSV`（支持线性功率或 `dB`/`dBm` 对数坐标），自动拟合洛伦兹峰、顺/逆时针背向散射劈裂双峰（Split-Doublet）、Fano 线型或多 FSR 宽谱，提取群折射率 $n_g$ 与波导损耗 $\alpha$（`dB/cm`），一键带入新微环版图综合。
4. **双向 GDSII / OASIS 版图桥（`上传 .gds 自动测量仿真` + `导出单器件/MPW 阵列 .gds`）**：
   - **上传 `.gds` 自动测量并仿真**：拖入任意由 KLayout 或 `gdsfactory` 绘制的 `.gds`/`.oas` 微环文件，内置几何引擎（`gdstk` + `shapely`）自动识别**圆环点耦合、跑道型、同心滑轮（Pulley）或双环游标（Vernier）**，直接从多边形精确量取 `R`、`W_ring`、`W_bus`、`gap_thru`、`gap_drop`、`L_s` 与 `pulley_angle`，在交互式 2D 画布上用红色卡尺标出耦合间距，确认膜厚后即可一键算出光谱！
   - **一键生成单器件或 MPW 扫参阵列 `.gds` + KLayout `.lyp`**：自动包含波导芯层（`1/0`）、半刻蚀平板层（`2/0`）、TiN 微加热器（`10/0`）、GSG 电光焊盘（`11/0`）、端口引脚（`65/0`）及 KLayout 配色文件（`.lyp`）。
5. **全波段覆盖（可见光 `400 nm` $\rightarrow$ O波段 `1310 nm` $\rightarrow$ C波段 `1550 nm` $\rightarrow$ 短波红外 `2 µm` $\rightarrow$ 中远红外 `3–14 µm`）× 7 大拓扑 × 24 种洁净室材料**：
   涵盖可见光/量子光学材料（`TiO2`、`Ta2O5`、`GaN`、`金刚石 Diamond`、`Al2O3`）、通信与非线性材料（`Si (SOI)`、`a-Si:H`、`LPCVD Si3N4`、`富硅氮化硅 SiNx`、`4H-SiC`、`3C-SiC`、`AlN`、`ScAlN (10%/20%/30%)`、`Z切/X切薄膜铌酸锂 TFLN`、`GSST 相变材料`）以及中红外低损耗材料与氟化物衬底（`Ge`、`CaF2`、`MgF2`、`Ge-on-Si`），内置跨波段自适应网格/仿真域缩放与带隙/多声子吸收自动预警（如 `Si < 1.12 µm` 禁带吸收与 `SiO2 > 3.6 µm` 中红外多声子吸收告警）。
6. **跨平台本地算力（macOS Apple Silicon / Windows CUDA GPU / 纯 CPU）且 100% MIT/BSD 开源合规**：
   全部求解在您自己的电脑本地离线完成，自动探测 **NVIDIA CUDA (`FP64`)**、**Apple Accelerate AMX / Metal** 或普通轻薄本的 **多核矢量化 CPU**；内置纯 BSD 约束 Delaunay 保形网格器（零 GPL 传染依赖）。

---

## 安装与运行：支持完全离线桌面软件（`.dmg`/`.exe`）或在线浏览器 / `Claude` / `Codex`

无需任何商业仿真软件许可证，您可自由选择**离线本地运行**或**在线云端 / AI 协同运行**：

### 方式一：像普通桌面软件一样离线安装（`.dmg` 与 `.exe` 免配置安装包，无需装 Python）

直接前往 [**GitHub Releases 发布页**](https://github.com/ZhihaoRenKevin/microring-studio/releases/latest) 下载对应系统的独立安装包：

| 操作系统 | 安装包文件 | 安装与首次打开方法 |
| :--- | :--- | :--- |
| **macOS（Apple Silicon）** | [**`MicroRing-Studio-v3.6.0-macOS-AppleSilicon.dmg`**](https://github.com/ZhihaoRenKevin/microring-studio/releases/latest) | 双击打开 `.dmg`，将 **`MicroRing Studio.app`** 拖入 **`Applications（应用程序）`**。*（首次打开开源未公证应用时，请右键点击 **`MicroRing Studio.app`** $\rightarrow$ 选择 **打开（Open）**，或在 **系统设置 $\rightarrow$ 隐私与安全性** 中点击“仍要打开”。）* |
| **Windows 10 / 11（64位）** | **`MicroRing-Studio-v3.6.0-Windows-Setup-x64.exe`** *（或免安装 `.zip`）* | 双击运行 **`Setup.exe`** 安装向导（无需管理员密码，自动创建桌面与开始菜单快捷方式）；也可按下方“方式三”下载源码 ZIP 后双击 `launchers\MicroRing_Windows.bat` 一键自动配置。 |

*（独立桌面版导出的 `.gds` 版图、`.lyp` 图层配色与 `.csv` 光谱默认保存在本地 `~/Documents/MicroRing_Studio/` 目录。）*

---

### 方式二：零安装在线网页运行 + 自由切换 `离线 / Claude / Codex` 模式

1. **浏览器一键在线运行（`GitHub Codespaces` 零本地安装）**：
   点击顶部绿色徽章 [**Run Online in GitHub Codespaces**](https://codespaces.new/ZhihaoRenKevin/microring-studio?quickstart=1)，GitHub 会自动启动云端容器并在网页中打开 MicroRing Studio 交互工作台（端口 `8765`），无需在本地安装任何环境。
2. **界面内一键切换离线与在线 AI 模式（`⚡ 离线引擎` | `🖥 本地 Ollama` | `🟣 Claude` | `🟢 Codex`）**：
   在左下角聊天输入框上方可随时切换 4 种模式：
   - **`⚡ 离线模式 · 内置物理引擎`（默认）**：100% 断网可用，零 Token 消耗。
   - **`🖥 离线模式 · 本地 Ollama / MLX`**：连接本机 `Ollama`（`127.0.0.1:11434`）或 `LM Studio`（`127.0.0.1:1234`），数据不出本机。
   - **`🟣 在线模式 · Claude` 与 `🟢 在线模式 · Codex / OpenAI`**：若电脑已登录 `claude`（Claude Code CLI）或 `codex`（OpenAI Codex CLI），**无需额外填写 API Key 即可直接调用订阅**；也可点击 `🔑 Key` 临时输入 API Key（仅保存在浏览器内存，关闭即清除）。所有 2D 麦克斯韦本征模、CMT 光谱与 `.GDS` 版图仍在本地精确计算，仅向大模型发送 `< 150 Tokens` 摘要！
3. **在 `Claude Code` 或 `Codex` 终端/桌面端直接调用（`MCP` 协议 `< 150 Tokens/次`）**：
   使用 `Claude Code` 或 `Codex` 打开项目文件夹，即可自动识别 [`AGENTS.md`](../AGENTS.md) 与 [`.mcp.json`](../.mcp.json)，直接用自然语言指挥 Agent 跑微环设计、正负胶剂量容差评估、扫参、光谱损耗拟合与 `.GDS` 生成。

---

### 方式三：下载源码 ZIP + 双击自动配置脚本（需已安装 Python 3.11+）

若电脑已安装 [Python 3.11 / 3.12 / 3.13](https://www.python.org/downloads/)（Windows 安装时请勾选 **`Add python.exe to PATH`**）：

1. 点击仓库主页右上角绿色按钮 **`<> Code`** $\rightarrow$ **`Download ZIP`** 并解压。
2. 打开 `launchers/` 目录，双击对应系统的启动脚本：
   - **Windows 10 / 11**：双击 **`launchers\MicroRing_Windows.bat`**
   - **macOS**：右键点击 **`launchers/MicroRing_macOS.command`** $\rightarrow$ **打开（Open）**
3. **首次运行（约 1 分钟）**会自动创建 `.venv` 并装好全部科学计算包，后续每次双击 **`< 2 秒`** 即可启动。

---

### 方式四：Python 开发者 / 脚本调用安装（`git` + `pip` 直接复制粘贴）

#### Windows 10 / 11（打开 `PowerShell` 或 `CMD`）
```powershell
git clone https://github.com/ZhihaoRenKevin/microring-studio.git
cd microring-studio

python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -e ".[desktop]"

# 启动桌面软件界面：
microring
```

#### macOS 或 Linux（打开终端 `Terminal`）
```bash
git clone https://github.com/ZhihaoRenKevin/microring-studio.git
cd microring-studio

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[desktop]"

# 启动桌面软件界面：
microring
```

> **提示（独立窗口 vs 浏览器模式）**：
> - 输入 `microring` 启动原生桌面窗口（若未装 `pywebview` 会自动回退至浏览器）；
> - 输入 `microring browser` 可直接在 Chrome / Edge / Safari 打开本地工作台 `http://127.0.0.1:8765`（完全本地离线，不上传任何数据）。

<details>
<summary><b>方式五：使用 Conda、<code>uv</code> 或启用 NVIDIA CUDA GPU 加速（点击展开）</b></summary>

#### 使用 Conda / Miniforge
```bash
conda create -n microring python=3.12 -y
conda activate microring
git clone https://github.com/ZhihaoRenKevin/microring-studio.git
cd microring-studio
pip install -e ".[desktop]"
microring
```

#### 使用 `uv`（极速安装及硬件加速扩展）
```bash
git clone https://github.com/ZhihaoRenKevin/microring-studio.git
cd microring-studio

# 普通轻薄本 CPU（Windows / macOS / Linux）：
uv sync --extra desktop

# NVIDIA RTX 显卡工作站（启用 FP64 CUDA 宽谱加速）：
uv sync --extra desktop --extra cuda

# Apple Silicon Mac（启用 AMX + Metal MLX 加速）：
uv sync --extra desktop --extra apple

uv run microring
```
</details>

---

### 可选：运行自检与循序渐进的 Cookbook 示例脚本

激活 `.venv`（或 Conda 环境）后即可直接运行：

```bash
# 0. 检查硬件加速后端、10万波长点 TMM 速度及 24 种材料折射率表：
python examples/00_getting_started/00_check_environment.py

# 1. 扫描 400 nm Si3N4 波导宽度，自动定位 TE0 单模截止宽度（TE1 出现点）：
python examples/00_getting_started/01_waveguide_dispersion_sweep.py

# 2. 设计 220 nm SOI 四端口跑道微环并执行耦合间距容差扫描：
python examples/01_devices_and_sweeps/01_soi_point_and_racetrack.py

# 3. 综合 4H-SiC 同心滑轮微环临界耦合包角（ER > 60 dB）与 X切铌酸锂电光调谐：
python examples/01_devices_and_sweeps/02_sin_and_sic_pulley_cc.py

# 3b. 跨波段示例（可见光 532/780 nm、O波段 1310 nm、短波红外 2 um、中红外 4.2/7.5 um）：
python examples/01_devices_and_sweeps/03_multiband_vis_1310_midir.py

# 3c. 正胶 vs 负胶光刻剂量过曝/欠曝分析与 5 单元预偏置补偿 .gds 导出：
python examples/01_devices_and_sweeps/04_lithography_dose_pr_tolerance.py

# 3d. 纯自研 2D-FEM 有源调谐 (硅 PN / Pockels EO / 热光加热器 + ΔT / E) 与 Lumerical INTERCONNECT 导出：
python examples/01_devices_and_sweeps/05_active_tuning_pn_eo_to_interconnect.py

# 3e. 零商业软件端面耦合器 (Butt)、二维全波 FDFD 光栅耦合器 (GC) 与超高 Q 收敛诊断：
python examples/01_devices_and_sweeps/06_butt_gc_couplers_and_high_q_fdtd.py

# 3f. 零 3D-FDTD 马赫-曾德尔干涉仪 (DC / MMI / Y分支)、推挽电光调制器与 INTERCONNECT 导出：
python examples/01_devices_and_sweeps/07_mzi_spectra_dc_mmi_and_modulators.py

# 4. 生成单微环版图及 5 器件 MPW 扫参阵列 .gds + KLayout .lyp：
python examples/02_gds_roundtrip/01_create_mrr_and_sweep_array_gds.py

# 5. 读取任意外部 .gds 版图 -> 自动提取几何参数 -> 求解模场与透射谱：
python examples/02_gds_roundtrip/02_upload_and_simulate_gds.py

# 6. 实验透射谱拟合（背向散射双峰劈裂 Q_int / Q_ext / 2g 与多 FSR 群折射率提取）：
python examples/03_lab_spectrum_fitting/01_fit_doublet_and_multifsr.py

# 运行全套 13 项自动化物理与接口验证测试：
python tests/test_microring_all.py
```

---

## 详细使用教学

### 1. 如何用左侧 Chatbot 从零设计微环并评估光刻过曝/欠曝容差？
打开软件后，左侧 **Photonics Design Copilot** 会引导你逐步确认需求（可直接点击气泡下方的快捷按钮，或输入中/英文自然语言）：
1. **输入材料与晶圆厚度**：例如输入 `600 nm 4H-SiC strip, SiO2 clad, TE0 at 1550 nm`（或 `400nm 氮化硅 条形波导 1550nm TE0`）。助手会自动调用 2D 有限元扫描波导宽度，给出单模最大宽度 `W_SM,max` 并填入推荐的 `W_bus` 和 `W_ring`。
2. **输入目标 FSR 与拓扑**：例如输入 `FSR 150 GHz, pulley all-pass`（或 `FSR 200 GHz 滑轮全通微环`）。助手会根据有限元算出的群折射率 $n_g$ 反解精确半径 $R$，完成全矢量弯曲模场与耦合积分，并给出**临界耦合目标值**。
3. **评估正/负胶剂量误差与导出补偿 GDS**：点击 `Apply Critical Coupling` 一键达到临界耦合，点击 **`🧪 工艺与剂量（Fab & Dose）`** 或直接输入 `检查正胶过曝欠曝` / `Check Negative PR dose error`，即可查看 `ER >= 15 dB` 剂量窗口、谐振漂移曲线与三拐角叠绘谱，并一键导出 `5-Cell Pre-Biased .GDS` 预偏置补偿阵列与 KLayout `.lyp`！

### 2. 如何上传已有 `.gds` 版图进行检查与光谱预测？
点击左侧聊天框顶部的 **`+ .GDS`**（或切到 **GDS 版图** 视图拖入 `.gds`/`.oas` 文件）：
- 右侧画布立即渲染分层多边形（支持滚轮缩放、拖拽平移、图层开关）并用红色标尺显示实测 `gap`；
- 聊天框列出自动识别的拓扑、半径、波导宽和耦合间距，点击对应的晶圆膜厚预设（如 `400 nm Si3N4 + Simulate GDS`）即可立刻仿真光谱。

### 3. 如何接入 `Codex`、`Claude Code` 与本地大模型（单次调用 `< 150 Tokens` + 100% 本地算力）？
仓库根目录内置 [`AGENTS.md`](../AGENTS.md) 与 [`.mcp.json`](../.mcp.json)，并提供原生 **MCP 协议服务端**（`uv run microring mcp`）及极简 **Agent 命令行**（`uv run microring agent ...`）：
- **重计算全部走本地 CPU / Apple AMX / NVIDIA CUDA + SSD 本征模缓存**：首次求解 2D FEM 后自动写入 `outputs/.mode_cache/`，后续间距/包角/剂量/温升扫参在本地 **`< 0.2 秒`** 完成，零云端算力开销；
- **拒绝向大模型倾倒长数组（节省 99% Token）**：完整的透射谱 `.csv` 与版图 `.gds`/`.lyp` 直接保存在本地 `outputs/agent_runs/`，返回给 Codex / Claude 的永远是 **`< 150 Tokens` 的单行精炼指标 JSON**（详见 [Codex 与 Claude 极省 Token 接入指南](./tutorials/ai_agents_codex_claude.md)）：

```bash
uv run microring agent design --top pulley_ap --mat 4H-SiC --h 0.60 --w-ring 1.10 --w-bus 0.82 --fsr 150 --gap 0.24 --cc --gds
uv run microring agent fab-tolerance --mat Si --h 0.22 --w-ring 0.48 --w-bus 0.45 --r 15 --gap 0.18 --resist positive_pr --dose 15 --bias-gds
```

---

## 作者、引用与许可证

- **作者**：**Zhihao Ren**（[ORCID: 0000-0002-2520-0784](https://orcid.org/0000-0002-2520-0784)）
- **学术引用**：若 MicroRing Studio 对您的科研、流片或教学有所帮助，欢迎通过 GitHub 右侧 *"Cite this repository"* 或 [CITATION.cff](../CITATION.cff) 引用本软件。
- **免责声明与许可证**：本仓库材料模型与默认层号源自公开学术文献，仅供光学设计、教学与流片前快速原型评估；正式流片前请务必使用目标代工厂官方 PDK 与 DRC 规则核验。本项目为独立开源项目，与 Ansys、Synopsys、COMSOL 无隶属或背书关系。代码遵循 [MIT 许可证](../LICENSE) 免费开源发布。
