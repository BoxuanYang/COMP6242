在环境和模型都已经配好的前提下，只跑 Self-Forcing mixed-bit，核心命令就是这些：

cd QuantVideoGen
# 确认当前 mixed-bit 配置
grep -n "mixed_bit_enabled\|mixed_schedule\|mixed_1bit_ratio\|mixed_low_quant_type\|mixed_high_quant_type\|num_output_frames\|local_attn_size" scripts/Self-Forcing/run_qvg.sh
# 创建日志目录
mkdir -p logs
# 运行 Self-Forcing mixed-bit 实验
bash scripts/Self-Forcing/run_qvg.sh 2>&1 | tee logs/self_forcing_mixed_bit.log
如果要改 1-bit 比例，先编辑：

vim scripts/Self-Forcing/run_qvg.sh
修改这一行：

mixed_1bit_ratio=0.25
然后重新跑：

bash scripts/Self-Forcing/run_qvg.sh 2>&1 | tee logs/self_forcing_mixed_bit_ratio_0.25.log
如果想快速检查输出：

find results/selfforcing -name "*.mp4"
如果想从日志里看 mixed-bit 分界和 KV cache 显存：

grep -E "Mixed-bit schedule|Mixed-bit KV spans|Total KV Cache Memory Usage|Per Layer Memory Usage|Quantization KV Cache Time" logs/self_forcing_mixed_bit.log
如果想跑多个比例，按现在脚本结构，最稳的是每次改 mixed_1bit_ratio 后分别跑：

# ratio = 0.25
bash scripts/Self-Forcing/run_qvg.sh 2>&1 | tee logs/self_forcing_mixed_bit_ratio_0.25.log
# ratio = 0.50
bash scripts/Self-Forcing/run_qvg.sh 2>&1 | tee logs/self_forcing_mixed_bit_ratio_0.50.log
# ratio = 0.75
bash scripts/Self-Forcing/run_qvg.sh 2>&1 | tee logs/self_forcing_mixed_bit_ratio_0.75.log