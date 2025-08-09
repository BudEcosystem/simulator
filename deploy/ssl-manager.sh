#!/bin/bash

# SSL Certificate Management Script

DOMAIN="simulator.bud.studio"
SSL_DIR="./ssl"
PROJECT_DIR="/home/budadmin/simulator"

echo "🔐 SSL Certificate Management for $DOMAIN"

# Check if certbot is installed
if ! command -v certbot &> /dev/null
then
    echo "❌ Certbot is not installed. Installing..."
    sudo apt-get update
    sudo apt-get install -y certbot
fi

echo "Select an option:"
echo "1. Obtain new certificates with Let's Encrypt"
echo "2. Renew existing certificates"
echo "3. Copy existing certificates to SSL directory"
echo "4. Test certificate configuration"

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "🔍 Obtaining new certificates..."
        sudo certbot certonly --standalone -d $DOMAIN
        
        if [ $? -eq 0 ]; then
            echo "✅ Certificates obtained successfully!"
            echo "📂 Copying certificates to $SSL_DIR..."
            mkdir -p $SSL_DIR
            sudo cp /etc/letsencrypt/live/$DOMAIN/fullchain.pem $SSL_DIR/
            sudo cp /etc/letsencrypt/live/$DOMAIN/privkey.pem $SSL_DIR/
            sudo chown $USER:$USER $SSL_DIR/*.pem
            echo "✅ Certificates copied to SSL directory!"
        else
            echo "❌ Failed to obtain certificates"
        fi
        ;;
    2)
        echo "🔄 Renewing certificates..."
        sudo certbot renew
        
        if [ $? -eq 0 ]; then
            echo "✅ Certificates renewed successfully!"
            echo "📂 Updating certificates in $SSL_DIR..."
            sudo cp /etc/letsencrypt/live/$DOMAIN/fullchain.pem $SSL_DIR/
            sudo cp /etc/letsencrypt/live/$DOMAIN/privkey.pem $SSL_DIR/
            sudo chown $USER:$USER $SSL_DIR/*.pem
            echo "✅ Certificates updated!"
        else
            echo "❌ Failed to renew certificates"
        fi
        ;;
    3)
        echo "📎 Copying existing certificates..."
        read -p "Enter path to fullchain.pem: " fullchain_path
        read -p "Enter path to privkey.pem: " privkey_path
        
        if [ -f "$fullchain_path" ] && [ -f "$privkey_path" ]; then
            mkdir -p $SSL_DIR
            cp "$fullchain_path" $SSL_DIR/fullchain.pem
            cp "$privkey_path" $SSL_DIR/privkey.pem
            echo "✅ Certificates copied to SSL directory!"
        else
            echo "❌ Certificate files not found"
        fi
        ;;
    4)
        echo "🧪 Testing certificate configuration..."
        if [ -f "$SSL_DIR/fullchain.pem" ] && [ -f "$SSL_DIR/privkey.pem" ]; then
            echo "✅ SSL certificates found in $SSL_DIR"
            echo "   - fullchain.pem: $(ls -lh $SSL_DIR/fullchain.pem | awk '{print $5}')"
            echo "   - privkey.pem: $(ls -lh $SSL_DIR/privkey.pem | awk '{print $5}')"
        else
            echo "❌ SSL certificates not found in $SSL_DIR"
            echo "   Please obtain certificates first"
        fi
        ;;
    *)
        echo "❌ Invalid choice"
        ;;
esac