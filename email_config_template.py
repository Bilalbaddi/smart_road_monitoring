# Email Configuration for Traffic Analysis System
# Replace these with your actual email credentials

def get_email_config():
    """
    Configure your email settings here.
    
    For Gmail:
    1. Enable 2-factor authentication
    2. Generate an app password: https://myaccount.google.com/apppasswords
    3. Use the app password instead of your regular password
    
    For other providers, update the SMTP settings accordingly.
    """
    
    # Example configurations for different email providers
    
    # GMAIL CONFIGURATION
    gmail_config = {
        'host': 'smtp.gmail.com',
        'port': 587,
        'user': 'your_email@gmail.com',  # Replace with your Gmail
        'password': 'your_app_password',  # Replace with your Gmail app password
        'to': 'recipient@gmail.com'       # Replace with recipient email
    }
    
    # OUTLOOK/HOTMAIL CONFIGURATION
    outlook_config = {
        'host': 'smtp-mail.outlook.com',
        'port': 587,
        'user': 'your_email@outlook.com',  # Replace with your Outlook email
        'password': 'your_password',        # Replace with your Outlook password
        'to': 'recipient@outlook.com'       # Replace with recipient email
    }
    
    # YAHOO CONFIGURATION
    yahoo_config = {
        'host': 'smtp.mail.yahoo.com',
        'port': 587,
        'user': 'your_email@yahoo.com',   # Replace with your Yahoo email
        'password': 'your_app_password',   # Replace with your Yahoo app password
        'to': 'recipient@yahoo.com'        # Replace with recipient email
    }
    
    # TEST CONFIGURATION (won't actually send emails)
    test_config = {
        'host': 'localhost',
        'port': 25,
        'user': 'test@test.com',
        'password': 'test123',
        'to': 'test@test.com'
    }
    
    # Return the configuration you want to use
    # Change this to gmail_config, outlook_config, yahoo_config, or your custom config
    return test_config  # Default to test config for safety

def validate_email_config(config):
    """Validate email configuration."""
    required_keys = ['host', 'port', 'user', 'password', 'to']
    for key in required_keys:
        if key not in config:
            return False, f"Missing required key: {key}"
        if not config[key]:
            return False, f"Empty value for key: {key}"
    return True, "Configuration valid"

# Quick setup instructions
SETUP_INSTRUCTIONS = """
EMAIL SETUP INSTRUCTIONS:

1. Choose your email provider (Gmail, Outlook, Yahoo, etc.)

2. For Gmail:
   - Go to https://myaccount.google.com/security
   - Enable 2-factor authentication
   - Go to https://myaccount.google.com/apppasswords
   - Generate an app password for "Mail"
   - Use this app password in the configuration

3. For Outlook/Hotmail:
   - Use your regular email and password
   - Make sure "Less secure app access" is enabled if needed

4. For Yahoo:
   - Go to Account Security settings
   - Generate an app password
   - Use this app password in the configuration

5. Edit the configuration in this file:
   - Replace 'your_email@gmail.com' with your actual email
   - Replace 'your_app_password' with your actual password/app password
   - Replace 'recipient@gmail.com' with the email where you want to receive alerts

6. Change the return statement in get_email_config() to use your provider:
   - return gmail_config
   - return outlook_config  
   - return yahoo_config

7. Test the configuration by running the system
"""

if __name__ == "__main__":
    print(SETUP_INSTRUCTIONS)
    config = get_email_config()
    is_valid, message = validate_email_config(config)
    print(f"\nCurrent configuration: {message}")
    print(f"Config: {config}")