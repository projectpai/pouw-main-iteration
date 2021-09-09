import subprocess
import pai.pouw.nodes.decentralized.worker


def call_wrapper(args):
    return call_worker(*args)


def call_worker(is_debug=False, cuda=False, use_paicoin=True, redis_host='localhost', redis_port=6379, python_interpreter='python3'):
    worker_script_path = pai.pouw.nodes.decentralized.worker.__file__
    # in order to ensure debugger is working properly
    if worker_script_path.endswith('pyc'):
        worker_script_path = worker_script_path[:-1]

    script_parameters = [python_interpreter, worker_script_path,
                         '--redis-host', redis_host,
                         '--redis-port', str(redis_port),
                         '--use-paicoin', str(use_paicoin),
                         '--cuda', str(cuda),
                         ]
    if is_debug:
        script_parameters.append('--debug')
        script_parameters.append('True')

    subprocess.call(
        script_parameters
    )