import evd

application = evd.server

def run_display(larndsim_dir, filepath="."):
    evd.run_display(larndsim_dir, filepath=filepath)

    return application

if __name__ == '__main__':
    application.run()