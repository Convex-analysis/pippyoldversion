# EVO-1模型训练代码修复计划

## 1. 模型架构修复
- **问题1**：`FlowmatchingActionHead` 类的 `__init__` 方法没有将 `config` 参数存储为实例变量
- **问题2**：`EVO1Driving` 类的 `forward` 方法调用 `self.action_head` 时参数名不匹配
- **问题3**：`compute_loss` 方法中 `_controls_to_waypoints` 逻辑不准确

## 2. 数据加载修复
- **问题1**：`_get_vehicle_state` 方法中实例变量未初始化
- **问题2**：`__getitem__` 方法中 `current_sample_token` 处理逻辑存在问题

## 3. 训练器修复
- **问题1**：`evaluate_global_model` 方法缺少 `F` 导入
- **问题2**：`setup_data_loader` 方法中 `client_id` 提取可能出错
- **问题3**：`load_checkpoint` 方法未恢复完整训练状态

## 4. 配置修复
- **问题1**：`TrainingConfig` 类缺少 `federated_learning` 参数

## 5. 训练脚本修复
- **问题1**：`project_root` 计算逻辑不准确
- **问题2**：`setup_wandb` 函数实现有问题

## 修复步骤
1. 修复模型架构文件 `model/evo1_driving.py`
2. 修复数据加载器 `data/nuscenes_loader.py`
3. 修复训练器 `training/federated_trainer.py`
4. 修复配置文件 `utils/config.py`
5. 修复训练脚本 `scripts/train_federated.py`
6. 验证修复后的代码能否正常执行

## 预期效果
- 代码语法错误修复
- 逻辑缺陷修复
- 资源引用问题解决
- 与FHDP框架兼容
- 能够正常执行完整的EVO-1模型训练流程