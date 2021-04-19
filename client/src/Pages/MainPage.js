import React, { useEffect, useRef, useState } from 'react'
import {
  ChakraProvider,
  VStack,
  Code,
  Grid,
  theme,
  Container,
  Center,
  Button,
  Heading,
  HStack
} from '@chakra-ui/react';
import axios from 'axios'
import ClassificationCard from '../components/ClassificationCard';

export default function MainPage() {

  const [topThree, setTopThree] = useState({})
  const [segmentImg, setSegmentImg] = useState()
  const [cardData, setCardData] = useState([])

  const [loading, setLoading] = useState(false)
  const url = "http://localhost:5000"

  const config = {
    headers: {
      "Content-Type": "application/json",
      'Access-Control-Allow-Origin': '*',
    }
  }

  const videoRef = useRef();

  useEffect(() => {
    navigator.mediaDevices
      .getUserMedia({
        video: true
      })
      .then((stream) => {
        videoRef.current.srcObject = stream
      })

    setInterval(() => {
      sendData()
    }, 2000)

  }, [])



  const sendData = () => {
    const canvas = document.createElement('canvas')
    // console.log(videoRef.current.videoHeight)
    // console.log(videoRef.current.videoWidth)
    canvas.height = videoRef.current.videoHeight - 90
    canvas.width = videoRef.current.videoWidth - 220
    const ctx = canvas.getContext('2d')
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height)
    const image = canvas.toDataURL()

    let data = { image: image.split(';base64,')[1] }
    axios.post(
      url + '/classify',
      data,
      config
    )
      .then((response) => {
        // console.log(response.data.classifications)
        setTopThree(response.data.classifications.map(l => String.fromCharCode(97 + l)))
      })
      .catch((err) => {
        console.log(err)
      })
  }


  const segment = () => {
    setLoading(true)
    const canvas = document.createElement('canvas')
    console.log(videoRef.current.videoHeight)
    console.log(videoRef.current.videoWidth)
    canvas.height = videoRef.current.videoHeight - 90
    canvas.width = videoRef.current.videoWidth - 220
    const ctx = canvas.getContext('2d')
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height)
    const image = canvas.toDataURL()

    let data = { image: image.split(';base64,')[1] }

    axios.post(
      url + '/segment',
      data,
      config
    )
      .then((response) => {
        console.log(response)
        // const blob = response.data.blob()
        // const url = URL.createObjectURL(blob)
        setSegmentImg(response.data)
        setLoading(false)
      })
      .catch((err) => {
        console.log(err)
      })
      setLoading(false)
  }

  const createCard = () => {
    let classification = topThree[0]
    let snapshot = videoRef.current
    setCardData([...cardData, {classification, pic: snapshot}])
  }

  const cardList = cardData.map((card, index) => <ClassificationCard key={index} card={card}/>)

  return (
    <>
    <Heading color="blue.600" fontWeight="bold" fontSize="50px" mb={10}> ASL Recognition </Heading>
      <Center mt={8} marginRight="120px" mb={0} pb={0}>
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
        <VStack pl="100px" pb="180px" spacing="50px">
          <Heading fontSize="45px">
            First Match: {topThree ? <span className="class-label"> {topThree[0]} </span> : ""}
          </Heading>
          <Heading fontSize="35px">
            Second Match: {topThree ? topThree[1] : ""}
          </Heading>
          <Heading fontSize="25px">
            Third Match: {topThree ? topThree[2] : ""}
          </Heading>

          <Button colorScheme="blue" onClick={() => createCard()} isLoading={loading}>
            Create Card for Boberto
          </Button>

        </VStack>
      </Center>
      
      {/* <Heading mb="50px"> Cards </Heading> */}
      <HStack overflowX="scroll" mb="100px" mx="100px" spacing="40px">
        {cardList}
      </HStack>
      {/* <img src={`data:image/png;base64,${segmentImg}`} /> */}

    </>
    // <Container textAlign="center">
    //     <Center my={8} marginRight="80px">

    //         
    //     </Center>



    // </Container>
  )
}
