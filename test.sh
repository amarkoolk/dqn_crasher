CERT1=/etc/ssl/certs/ca-bundle.crt
CERT2=/etc/ssl/certs/ca-certificates.crt
if test -f "$CERT1"; then
    CERT=$CERT1
elif test -f "$CERT2"; then
    CERT=$CERT2
fi
echo $CERT
