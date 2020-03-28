import traceback
from multiprocessing import Process, Pipe

from joblib import Parallel, delayed


def parallel_async_invoke(payloads, work_func):
    # => https://aws.amazon.com/blogs/compute/parallel-processing-in-python-with-aws-lambda/
    # create a list to keep all processes
    processes = []

    # create a list to keep connections
    parent_connections = []
    
    for input_payload in payloads:            
        parent_conn, child_conn = Pipe()
        parent_connections.append(parent_conn)

        # create the process, pass instance and connection
        process = Process(target=work_func, args=(input_payload, child_conn,))
        processes.append(process)

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    results = []
    for parent_connection in parent_connections:
        results.append(parent_connection.recv()[0])

    return results


def joblib_parallel(payloads, workfunc):
    return Parallel(n_jobs=4, verbose=10
            )(delayed(workfunc)(input_payload)
                    for input_payload in payloads)
    
