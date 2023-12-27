import paramiko
import os

supported_languages = ('hin', 'eng', 'chi', 'ger', 'ben', 'tam')

def create_image_remotely(text, lang):
    
    if lang not in supported_languages:
        raise Exception("lang not supported")
    
    # SSH settings for machine A
    host = '172.25.0.208'  # Replace with the IP address or hostname of machine A
    cred = 'iit_roorkee_tcp'  # Replace with your username on machine A
    main_dir = '/Data/Onkar'

    # Connect to machine A
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, username=cred, password=cred)

    # Execute the create_image function remotely
    remote_command = f'python /DATA/onkar/pg/synth_vistrans/create_input_text/demo/create_image2.py {text} {lang}'
    stdin, stdout, stderr = ssh.exec_command(remote_command)
    
    # Wait for the command to finish (optional)
    stdout.channel.recv_exit_status()

    # Close the SSH connection
    ssh.close()

    # Transfer the image from machine A to machine B
    local_filename = 'i_t.png'
    remote_filename = '/DATA/onkar/synth_vistrans/create_input_text/demo/i_t.png'  # Replace with the path of the generated image on machine A

    with paramiko.Transport((host, 22)) as transport:
        transport.connect(username=cred, password=cred)
        sftp = paramiko.SFTPClient.from_transport(transport)
        sftp.get(remote_filename, local_filename)

if __name__ == '__main__':
    create_image_remotely('test', 'eng')
