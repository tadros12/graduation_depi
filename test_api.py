import os
import subprocess
import time
import urllib.request
import urllib.error
import json

def test_api():
    print("Starting FastAPI server on port 8085 for verification...")
    # Start the server in the deployment folder
    server_proc = subprocess.Popen(
        ["/home/theodoros/graduation/.venv/bin/python", "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8085"],
        cwd="/run/media/theodoros/E/projects/dataflow__analyizer/depi_project/deployment",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Wait for the server to spin up
    time.sleep(4)

    try:
        url = "http://127.0.0.1:8085/api/predict"
        # Use one of the Alexandria images from the dataset
        filepath = "/run/media/theodoros/E/projects/dataflow__analyizer/depi_project/egypt_s2_diverse_dataset/Alexandria_000_29.959_31.183.tif"
        
        if not os.path.exists(filepath):
            print(f"Error: Sample file not found at {filepath}")
            return

        # Prepare multipart/form-data payload
        boundary = "----WebKitFormBoundaryTestRequest"
        with open(filepath, "rb") as f:
            file_content = f.read()
            
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{os.path.basename(filepath)}"\r\n'
            f"Content-Type: image/tiff\r\n\r\n"
        ).encode('utf-8') + file_content + f"\r\n--{boundary}--\r\n".encode('utf-8')
        
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body))
            },
            method="POST"
        )
        
        print("Sending sample TIFF file to /api/predict...")
        with urllib.request.urlopen(req) as response:
            resp_data = json.loads(response.read().decode('utf-8'))
            print("\n✅ API Validation Succeeded! Response details:")
            print(f"  Filename: {resp_data.get('filename')}")
            print(f"  Dominant Class: {resp_data.get('dominant_class')}")
            print(f"  Classes Detected: {resp_data.get('classes_detected')}")
            print(f"  Total Pixels: {resp_data.get('total_pixels')}")
            print(f"  Percentages: {resp_data.get('percentages')}")
            print(f"  Charts Generated: {list(resp_data.get('charts', {}).keys())}")
            
    except urllib.error.HTTPError as he:
        print(f"\n❌ API Validation FAILED with HTTPError {he.code}:")
        try:
            error_body = he.read().decode('utf-8')
            print(f"  Error Response Body: {error_body}")
        except Exception as read_err:
            print(f"  Could not read error response: {read_err}")
            
    except Exception as e:
        print(f"\n❌ API Validation FAILED: {e}")
        # Print server output
        print("\n--- Server Output ---")
        try:
            # Terminate the server first so we can read its output
            server_proc.terminate()
            stdout_data, stderr_data = server_proc.communicate(timeout=5)
            print("Stdout:")
            print(stdout_data.decode('utf-8', errors='ignore'))
            print("Stderr:")
            print(stderr_data.decode('utf-8', errors='ignore'))
        except Exception as read_err:
            print(f"Could not read server logs: {read_err}")
        print("---------------------")
        
    finally:
        # Check if still running (in case terminate failed)
        if server_proc.poll() is None:
            print("Stopping verification server...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()
            print("Server stopped cleanly.")
        else:
            print("Server already stopped.")

if __name__ == "__main__":
    test_api()
