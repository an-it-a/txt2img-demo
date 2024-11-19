# **Cloud Run txt-2-img DEMO**

<img src="https://iam-an-it-a.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F9a1fc3e9-77f4-4f8f-99fb-cc93639e3f8a%2F6b910585-d14f-4638-bba7-5a66af6004fe%2F30f5b391-9eca-4132-93f4-859727973b2e.png?table=block&id=13f2ff85-3f4f-80b3-bc2b-cd93da2b10e0&spaceId=9a1fc3e9-77f4-4f8f-99fb-cc93639e3f8a&width=1420&userId=&cache=v2" width="20"/>[Youtube](http://www.youtube.com/@an.it.a)
<img src="https://iam-an-it-a.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F9a1fc3e9-77f4-4f8f-99fb-cc93639e3f8a%2F3bcfcd02-dd89-476f-b77f-1a94dde4f9d1%2Fimage.png?table=block&id=13f2ff85-3f4f-80d4-bcf4-ec049cbd7bcc&spaceId=9a1fc3e9-77f4-4f8f-99fb-cc93639e3f8a&width=1200&userId=&cache=v2" width="20"/>[Instagram](https://instagram.com/iam.an.it.a)<br/>
Subscribe and Follow me! 💪🏻<br/>

<hr/>

## Cloud storage
This project requires 1 Cloud Storage bucket to store the output results
### Folders:
1. output
    * output results will be stored here

## Cloud Run Setup
1. GPU enabled
2. Volume Mount
   * Volume type: `Cloud Storage Bucket` 
   * Mount Path: `/vol`

## Request Curl
<i>Remember to replace your endpoint URL!</i>
```
curl -X POST -H 'Content-Type: application/json' \
-H "Authorization: Bearer $(gcloud auth print-identity-token)" \
-d '{"prompt":"1girl, long blonde hair"}' \
https://txt2img-demo-111919368185.us-central1.run.app | jq .
```

<hr/>
