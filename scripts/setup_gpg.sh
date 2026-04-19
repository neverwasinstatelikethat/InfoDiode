#!/bin/bash
# Настройка GPG-ключа для шифрования XML

set -e

GPG_HOME="${INFODIODE_ENCRYPTION_KEYS_DIR:-/app/data/encryption_keys}"
GPG_RECIPIENT="${INFODIODE_GPG_RECIPIENT:-infodiode@local}"

mkdir -p "$GPG_HOME"
chmod 700 "$GPG_HOME"

# Проверяем, существует ли уже ключ
if gpg --homedir "$GPG_HOME" --list-keys "$GPG_RECIPIENT" >/dev/null 2>&1; then
    echo "GPG-ключ для $GPG_RECIPIENT уже существует"
    exit 0
fi

# Генерируем новый ключ (без пароля, для автоматизации)
cat > /tmp/gpg-batch <<EOF
%no-protection
Key-Type: RSA
Key-Length: 4096
Subkey-Type: RSA
Subkey-Length: 4096
Name-Real: InfoDiode
Name-Email: $GPG_RECIPIENT
Expire-Date: 0
EOF

gpg --homedir "$GPG_HOME" --batch --gen-key /tmp/gpg-batch
rm -f /tmp/gpg-batch

echo "GPG-ключ успешно создан для $GPG_RECIPIENT"
