# **Cloud Run txt-2-img DEMO**

<img src="https://assets.littlewitchanita.workers.dev/youtube-app-icon.png" width="20"/>[Youtube](http://www.youtube.com/@an.it.a)&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://assets.littlewitchanita.workers.dev/threads-app-icon.png" width="20"/>[Threads](https://www.threads.net/@iam.an.it.a)&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://assets.littlewitchanita.workers.dev/ig-instagram-icon.png" width="20"/>[Instagram](https://instagram.com/iam.an.it.a)&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://assets.littlewitchanita.workers.dev/linkedin-app-icon.png" width="20"/>[LinkedIn](https://www.linkedin.com/in/anita-wong-4ba01226)<br/>
Subscribe and Follow me! 💪🏻<br/>

---

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
-d '{"prompt":"1girl, long blonde hair, facing camera, smiling, professionally, professionally color graded, half body shot, sharp focus, 8 k high definition, dslr, soft lighting, insanely detailed, intricate, elegant, matte skin"}' \
https://txt2img-demo-111919368185.us-central1.run.app | jq .
```

<hr/>
