[app]
title = QuantumShield Mobile
package.name = quantumshieldmobile
package.domain = org.quantumshield

source.dir = .
source.include_exts = py,kv,json,png,jpg,ttf

version = 0.1.0

requirements = python3,kivy,numpy,requests

orientation = portrait
fullscreen = 0

android.permissions = INTERNET

android.api = 34
android.minapi = 26
android.ndk = 25b

android.archs = arm64-v8a

android.enable_androidx = True

[buildozer]
log_level = 2
warn_on_root = 1

