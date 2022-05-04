import paramiko

ip = "172.30.1.40"
try:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    ssh.connect(ip, username="inclab", password="1q2w3e4r0110!")
    print("{} is connected".format(ip))

    channel = ssh_client.invoke_shell()
    channel.send('sudo su -\n')
    outdata, errdata = waitStrems(channel)
    print(outdata)

    channel.send('1q2w3e4r0110!\n')
    outdata, errdata = waitStrems(channel)
    print(outdata)

    channel.send('whoami\n')
    outdata, errdata = waitStrems(channel)
    print(outdata)

except Exception as err:
    print(err)
