import paramiko

ip = "172.30.1.40"
try:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    ssh.connect(ip, username="inclab", password="1q2w3e4r0110!")
    print("{} is connected".format(ip))
    stdin, stdout, dtderr = ssh.exec_command("ls -l")

    lines = stdout.readlines()
    for i in lines:
        re = str(i).replace('\n','')
        print(re)
    ssh.close()

except Exception as err:
    print(err)
