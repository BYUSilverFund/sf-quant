from sf_quant.backtester.slurm import BacktestDynamicConfig, BacktestDynamicRunner, SlurmConfig

slurm_config = SlurmConfig(
    n_cpus=8,
    mem="32G",
    time="03:00:00",
    mail_type="BEGIN,END,FAIL",
    max_concurrent_jobs=30,
)

backtest_config = BacktestDynamicConfig(
    signal_name='momentum',
    data_path='momentum_alphas.parquet',
    initial_gamma=50,
    target_active_risk=0.05,
    active_weights=True,
    project_root="/home/amh1124/Projects/sf-quant",
    byu_email="amh1124@byu.edu",
    constraints=[
        "ZeroBeta",
        "ZeroInvestment"
    ],
    slurm=slurm_config,
)

backtest_runner = BacktestDynamicRunner(backtest_config)
backtest_runner.submit()