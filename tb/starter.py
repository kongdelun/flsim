from tensorboard import program
from env import TB_OUTPUT

if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', TB_OUTPUT])
    tb.main()
