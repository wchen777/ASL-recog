import React, { useEffect, useRef } from 'react'
import {
    ChakraProvider,
    Box,
    Text,
    Link,
    VStack,
    Code,
    Grid,
    theme,
    Container,
    Center,
    Button
} from '@chakra-ui/react';
import axios from 'axios'

export default function MainPage() {
    const videoRef = useRef();
    //good
    useEffect(() => {
        navigator.mediaDevices
            .getUserMedia({
                video: true
            })
            .then((stream) => {
                videoRef.current.srcObject = stream
            })
        const canvas = document.createElement('canvas')
        // console.log(videoRef.current.videoHeight)
        // console.log(videoRef.current.videoWidth)
        canvas.height = videoRef.current.videoHeight
        canvas.width = videoRef.current.videoWidth
        const ctx = canvas.getContext('2d')
        // setInterval(() => {
        //     ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height)
        // })


    }, [])



    const sendData = () => {
        const canvas = document.createElement('canvas')
        console.log(videoRef.current.videoHeight)
        console.log(videoRef.current.videoWidth)
        canvas.height = videoRef.current.videoHeight - 90
        canvas.width = videoRef.current.videoWidth - 220
        // canvas.height = 600
        // canvas.width = 600
        const ctx = canvas.getContext('2d')
        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height)
        const image = canvas.toDataURL()
        console.log(image)


    }


    return (
        <>
            <Center my={8} marginRight="120px">
                <div className="container">
                    <video
                        id="vid"
                        width="600"
                        height="600"
                        autoPlay
                        ref={videoRef}
                    // style={{ width: 600 }}
                    ></video>
                </div>
            </Center>
            <Button colorScheme="blue" onClick={() => sendData()}>
                Boberto
            </Button>

        </>
        // <Container textAlign="center">
        //     <Center my={8} marginRight="80px">

        //         
        //     </Center>



        // </Container>
    )
}
