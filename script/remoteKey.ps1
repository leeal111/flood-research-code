# 为远程服务器注册SSH公钥，一般只使用一次

$remoteHostName = "172.17.0.2"
$remoteUserName = "root"
$sshFilePath = "~/.ssh/id_rsa.pub"

route add 172.17.0.0 mask 255.255.255.0 172.16.255.1
function ssh-copy-id([string]$userAtMachine, $args) {   
    $publicKey = "$ENV:USERPROFILE" + "/.ssh/id_rsa.pub"
    if (!(Test-Path "$publicKey")) {
        Write-Error "ERROR: failed to open ID file '$publicKey': No such file"            
    }
    else {
        & cat "$publicKey" | ssh $args $userAtMachine "umask 077; test -d .ssh || mkdir .ssh ; cat >> .ssh/authorized_keys || exit 1"      
    }
}
ssh-copy-id -i $sshFilePath $remoteUserName@$remoteHostName