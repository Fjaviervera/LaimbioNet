import tensorflow as tf


def launch_tensorboard(logs_path, port = '6007'):
    from tensorboard import default
    from tensorboard import program

    tb = program.TensorBoard()
    tb.configure(argv=['--logdir ' + logs_path])
    # tb.configure(argv=['--port ' + port])

    tb.main()


