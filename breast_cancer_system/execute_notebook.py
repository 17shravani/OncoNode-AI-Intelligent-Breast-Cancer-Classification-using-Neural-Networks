import json
import io
import sys
import os
import base64
import traceback
import contextlib
import matplotlib.pyplot as plt

def execute_notebook(notebook_path, output_path):
    print(f"Reading notebook from {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
        
    global_env = {}
    # Preset environment variables if needed
    global_env['__name__'] = '__main__'
    
    execution_count = 1
    
    for idx, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            print(f"Executing cell {execution_count}...")
            code_lines = cell['source']
            code_text = "".join(code_lines)
            
            # Reset cell outputs and execution count
            cell['outputs'] = []
            cell['execution_count'] = execution_count
            
            # Capture stdout and stderr
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            
            # Close any previous plots to avoid leak
            plt.close('all')
            
            success = True
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                try:
                    # Execute the code block in the shared global environment
                    exec(code_text, global_env)
                except Exception as e:
                    success = False
                    traceback.print_exc(file=sys.stderr)
            
            # Process stdout
            stdout_text = stdout_buf.getvalue()
            if stdout_text:
                cell['outputs'].append({
                    "output_type": "stream",
                    "name": "stdout",
                    "text": [line + '\n' for line in stdout_text.splitlines()]
                })
                
            # Process stderr / Traceback
            stderr_text = stderr_buf.getvalue()
            if stderr_text:
                cell['outputs'].append({
                    "output_type": "stream",
                    "name": "stderr",
                    "text": [line + '\n' for line in stderr_text.splitlines()]
                })
                
            # Capture any matplotlib figures created in this cell
            fignums = plt.get_fignums()
            if fignums:
                for fignum in fignums:
                    fig = plt.figure(fignum)
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                    buf.seek(0)
                    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                    
                    cell['outputs'].append({
                        "output_type": "display_data",
                        "data": {
                            "image/png": img_base64,
                            "text/plain": [f"<Figure size {fig.get_size_inches()[0]*100}x{fig.get_size_inches()[1]*100} with {len(fig.axes)} Axes>"]
                        },
                        "metadata": {}
                    })
                plt.close('all')
                
            if not success:
                print(f"Cell {execution_count} failed to execute! Check standard error in output.")
                
            execution_count += 1
            
    print(f"Writing executed notebook to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    print("Notebook executed and saved successfully!")

if __name__ == '__main__':
    notebook_file = "breast_cancer_system/Breast_Cancer_Classification_Professional.ipynb"
    execute_notebook(notebook_file, notebook_file)
    
    target_file = "Cancer Classification using neural Networks  June 2026/Cancer Classification using neural Networks/Breast_Cancer_Classification_with_Neural_Network.ipynb"
    if os.path.exists(target_file):
        execute_notebook(target_file, target_file)
