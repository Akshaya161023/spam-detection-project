import pandas as pd
import os

# ── Load original dataset safely ──────────────────────────────
print("Loading original spam.csv...")
try:
    df1 = pd.read_csv("data/spam.csv", encoding="latin-1")
except UnicodeDecodeError:
    df1 = pd.read_csv("data/spam.csv", encoding="utf-8")

# Keep only label + message columns
df1 = df1[["label", "message"]].copy()
df1 = df1.dropna(subset=["message"])
df1["message"] = df1["message"].astype(str)
print(f"Original dataset: {len(df1)} rows")
print(df1["label"].value_counts())

# ── Phishing + formal email patterns ─────────────────────────
extra = [
    # Phishing spam
    ("spam", "Dear User your account has been suspended please verify your details immediately at http://secure-login.xyz"),
    ("spam", "Your subscription will be cancelled today due to billing issue update your payment details using secure link http://update-payment-now.xyz"),
    ("spam", "Dear Customer your bank account has been locked click here to unlock http://bankverify.xyz"),
    ("spam", "Action required your PayPal account will be limited unless you confirm your information now"),
    ("spam", "Security alert we detected unusual login to your account verify immediately or account will be closed"),
    ("spam", "Dear valued customer your credit card has been charged update billing info to avoid service interruption"),
    ("spam", "Your Apple ID has been disabled verify your account within 24 hours to restore access"),
    ("spam", "We noticed suspicious activity on your account please login to verify your identity immediately"),
    ("spam", "Dear user your password will expire today click here to reset your password now"),
    ("spam", "Urgent your account verification is pending failure to verify will result in permanent suspension"),
    ("spam", "Your Amazon order has been placed click here to cancel if this was not you"),
    ("spam", "Dear client we are unable to process your recent payment please update your billing information"),
    ("spam", "You have a pending package delivery click here to confirm your address and pay delivery fee"),
    ("spam", "Your Netflix account will be cancelled update payment method immediately to continue service"),
    ("spam", "Tax refund notification you are eligible for refund click here to claim your money now"),
    ("spam", "Job offer work from home earn 50000 per month no experience required apply now"),
    ("spam", "Loan approved instant loan approved for you click here to get money transferred now"),
    ("spam", "Dear user your KYC verification is pending complete now or your account will be blocked"),
    ("spam", "Investment opportunity guaranteed 40 percent returns monthly invest now limited slots available"),
    ("spam", "Warning your computer is infected with virus call our toll free number immediately for support"),
    ("spam", "Dear beneficiary your unclaimed funds awaiting transfer send your bank details to claim"),
    ("spam", "You won lottery prize send your bank details to collect your prize money today"),
    ("spam", "Make money online guaranteed income working from home register now free training provided"),
    ("spam", "Your parcel is on hold customs fee required pay now to release your package delivery"),
    ("spam", "Final notice your electricity will be disconnected in 2 hours pay immediately to avoid disconnection"),
    ("spam", "Congratulations you have been selected for special reward claim your gift card now limited time"),
    ("spam", "Dear account holder unusual activity detected on your account please verify identity immediately"),
    ("spam", "Your OTP is 4521 do not share this OTP with anyone your bank never asks for OTP"),
    ("spam", "Exclusive offer upgrade now and get 6 months free click link below to activate offer"),
    ("spam", "Cheap medication available without prescription order now discreet worldwide shipping guaranteed"),
    # Formal ham emails
    ("ham", "Dear User please find attached the meeting agenda for tomorrow kindly review before the session"),
    ("ham", "Hi team the quarterly report is ready for review please check the attached document"),
    ("ham", "Your order has been shipped and will arrive by Friday please check your email for tracking"),
    ("ham", "Dear customer thank you for your payment your subscription has been renewed successfully"),
    ("ham", "Reminder your appointment is scheduled for tomorrow at 10am please confirm your attendance"),
    ("ham", "Dear student your assignment submission deadline is next Monday please submit via the portal"),
    ("ham", "Thank you for contacting support your ticket has been received we will respond within 24 hours"),
    ("ham", "Hi please find the invoice for last month services attached kindly process payment at earliest"),
    ("ham", "Your password was changed successfully if you did not make this change please contact support"),
    ("ham", "Dear applicant we have received your application and will get back to you within 5 business days"),
    ("ham", "Meeting rescheduled to 3pm on Wednesday please update your calendar accordingly"),
    ("ham", "Your subscription renewal is coming up no action needed we will auto renew your plan"),
    ("ham", "Please review and sign the attached document at your earliest convenience thank you"),
    ("ham", "Hi the project deadline has been extended to next Friday please update your tasks accordingly"),
    ("ham", "Your account statement for March is now available please login to view your statement online"),
]

df2 = pd.DataFrame(extra, columns=["label", "message"])
df2["message"] = df2["message"].astype(str)
print(f"\nExtra patterns: {len(df2)} rows")

# ── Combine safely ────────────────────────────────────────────
df_combined = pd.concat([df1, df2], ignore_index=True)
df_combined = df_combined.dropna(subset=["message", "label"])
df_combined = df_combined[df_combined["message"].str.strip() != ""]
df_combined = df_combined.drop_duplicates(subset=["message"])
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

df_combined.to_csv("data/spam.csv", index=False, encoding="utf-8")

print(f"\nFinal combined dataset:")
print(f"Total rows : {len(df_combined)}")
print(df_combined["label"].value_counts())
print("\nDone — data/spam.csv updated.")