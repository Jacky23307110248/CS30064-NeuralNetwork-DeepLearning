"""
Legacy entry point — use main.py for training.

  python main.py compare     # bn_compare (2 runs)
  python main.py landscape   # loss_landscape (8 runs)
  python main.py grad        # grad_probe distance sweep (2 runs)
  python main.py all         # compare + landscape (10 runs)
"""
if __name__ == "__main__":
    import main as cli
    import sys
    sys.argv = [sys.argv[0], "all"] if len(sys.argv) == 1 else sys.argv
    cli.main()
